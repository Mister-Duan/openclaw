import {
  resolveAgentDir,
  resolveAgentWorkspaceDir,
  resolveSessionAgentId,
  resolveAgentSkillsFilter,
} from "../../agents/agent-scope.js";
import { resolveModelRefFromString } from "../../agents/model-selection.js";
import { resolveAgentTimeoutMs } from "../../agents/timeout.js";
import { DEFAULT_AGENT_WORKSPACE_DIR, ensureAgentWorkspace } from "../../agents/workspace.js";
import { resolveChannelModelOverride } from "../../channels/model-overrides.js";
import { type OpenClawConfig, loadConfig } from "../../config/config.js";
import { applyLinkUnderstanding } from "../../link-understanding/apply.js";
import { applyMediaUnderstanding } from "../../media-understanding/apply.js";
import { defaultRuntime } from "../../runtime.js";
import { resolveCommandAuthorization } from "../command-auth.js";
import type { MsgContext } from "../templating.js";
import { SILENT_REPLY_TOKEN } from "../tokens.js";
import type { GetReplyOptions, ReplyPayload } from "../types.js";
import { emitResetCommandHooks, type ResetCommandAction } from "./commands-core.js";
import { resolveDefaultModel } from "./directive-handling.js";
import { resolveReplyDirectives } from "./get-reply-directives.js";
import { handleInlineActions } from "./get-reply-inline-actions.js";
import { runPreparedReply } from "./get-reply-run.js";
import { finalizeInboundContext } from "./inbound-context.js";
import { emitPreAgentMessageHooks } from "./message-preprocess-hooks.js";
import { applyResetModelOverride } from "./session-reset-model.js";
import { initSessionState } from "./session.js";
import { stageSandboxMedia } from "./stage-sandbox-media.js";
import { createTypingController } from "./typing.js";

function mergeSkillFilters(channelFilter?: string[], agentFilter?: string[]): string[] | undefined {
  const normalize = (list?: string[]) => {
    if (!Array.isArray(list)) {
      return undefined;
    }
    return list.map((entry) => String(entry).trim()).filter(Boolean);
  };
  const channel = normalize(channelFilter);
  const agent = normalize(agentFilter);
  if (!channel && !agent) {
    return undefined;
  }
  if (!channel) {
    return agent;
  }
  if (!agent) {
    return channel;
  }
  if (channel.length === 0 || agent.length === 0) {
    return [];
  }
  const agentSet = new Set(agent);
  return channel.filter((name) => agentSet.has(name));
}

/**
 * 自动回复系统的核心入口函数，负责处理收到的消息并生成 AI 回复。
 * 主要阶段：配置解析 → 消息预处理 → 会话状态初始化 → 指令解析 → 内联动作处理 → AI 回复生成
 */
export async function getReplyFromConfig(
  ctx: MsgContext,
  opts?: GetReplyOptions,
  configOverride?: OpenClawConfig,
): Promise<ReplyPayload | ReplyPayload[] | undefined> {
  // ─────────────────────────────────────────────────────────────────────────────
  // 阶段 1: 初始化与配置解析
  // - 加载配置、解析 agent ID 和会话 key
  // - 合并技能过滤器（channel + agent 级别）
  // - 解析默认模型；若是心跳消息则使用心跳专用模型
  // ─────────────────────────────────────────────────────────────────────────────
  const isFastTestEnv = process.env.OPENCLAW_TEST_FAST === "1";
  const cfg = configOverride ?? loadConfig();
  const targetSessionKey =
    ctx.CommandSource === "native" ? ctx.CommandTargetSessionKey?.trim() : undefined;
  const agentSessionKey = targetSessionKey || ctx.SessionKey;
  const agentId = resolveSessionAgentId({
    sessionKey: agentSessionKey,
    config: cfg,
  });
  const mergedSkillFilter = mergeSkillFilters(
    opts?.skillFilter,
    resolveAgentSkillsFilter(cfg, agentId),
  );
  const resolvedOpts =
    mergedSkillFilter !== undefined ? { ...opts, skillFilter: mergedSkillFilter } : opts;
  const agentCfg = cfg.agents?.defaults;
  const sessionCfg = cfg.session;
  const { defaultProvider, defaultModel, aliasIndex } = resolveDefaultModel({
    cfg,
    agentId,
  });
  let provider = defaultProvider;
  let model = defaultModel;
  let hasResolvedHeartbeatModelOverride = false;
  if (opts?.isHeartbeat) {
    // Prefer the resolved per-agent heartbeat model passed from the heartbeat runner,
    // fall back to the global defaults heartbeat model for backward compatibility.
    const heartbeatRaw =
      opts.heartbeatModelOverride?.trim() ?? agentCfg?.heartbeat?.model?.trim() ?? "";
    const heartbeatRef = heartbeatRaw
      ? resolveModelRefFromString({
          raw: heartbeatRaw,
          defaultProvider,
          aliasIndex,
        })
      : null;
    if (heartbeatRef) {
      provider = heartbeatRef.ref.provider;
      model = heartbeatRef.ref.model;
      hasResolvedHeartbeatModelOverride = true;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // 阶段 2: 工作区与超时设置
  // - 确保 agent 工作区目录存在，创建引导文件
  // - 解析 agent 目录和超时时间
  // - 创建打字状态控制器（模拟"正在输入..."）
  // ─────────────────────────────────────────────────────────────────────────────
  const workspaceDirRaw = resolveAgentWorkspaceDir(cfg, agentId) ?? DEFAULT_AGENT_WORKSPACE_DIR;
  const workspace = await ensureAgentWorkspace({
    dir: workspaceDirRaw,
    ensureBootstrapFiles: !agentCfg?.skipBootstrap && !isFastTestEnv,
  });
  const workspaceDir = workspace.dir;
  const agentDir = resolveAgentDir(cfg, agentId);
  const timeoutMs = resolveAgentTimeoutMs({ cfg, overrideSeconds: opts?.timeoutOverrideSeconds });
  const configuredTypingSeconds =
    agentCfg?.typingIntervalSeconds ?? sessionCfg?.typingIntervalSeconds;
  const typingIntervalSeconds =
    typeof configuredTypingSeconds === "number" ? configuredTypingSeconds : 6;
  const typing = createTypingController({
    onReplyStart: opts?.onReplyStart,
    onCleanup: opts?.onTypingCleanup,
    typingIntervalSeconds,
    silentToken: SILENT_REPLY_TOKEN,
    log: defaultRuntime.log,
  });
  opts?.onTypingController?.(typing);

  // ─────────────────────────────────────────────────────────────────────────────
  // 阶段 3: 消息预处理
  // - 最终化入站上下文
  // - 应用媒体理解（图片/视频分析）
  // - 应用链接理解（URL 内容提取）
  // - 触发消息预处理钩子
  // ─────────────────────────────────────────────────────────────────────────────
  const finalized = finalizeInboundContext(ctx);

  if (!isFastTestEnv) {
    await applyMediaUnderstanding({
      ctx: finalized,
      cfg,
      agentDir,
      activeModel: { provider, model },
    });
    await applyLinkUnderstanding({
      ctx: finalized,
      cfg,
    });
  }
  emitPreAgentMessageHooks({
    ctx: finalized,
    cfg,
    isFastTestEnv,
  });

  // ─────────────────────────────────────────────────────────────────────────────
  // 阶段 4: 会话状态初始化
  // - 解析命令授权
  // - 初始化会话状态（加载历史、检测重置、解析分组）
  // - 应用重置时的模型覆盖
  // ─────────────────────────────────────────────────────────────────────────────
  const commandAuthorized = finalized.CommandAuthorized;
  resolveCommandAuthorization({
    ctx: finalized,
    cfg,
    commandAuthorized,
  });
  const sessionState = await initSessionState({
    ctx: finalized,
    cfg,
    commandAuthorized,
  });
  let {
    sessionCtx,
    sessionEntry,
    previousSessionEntry,
    sessionStore,
    sessionKey,
    sessionId,
    isNewSession,
    resetTriggered,
    systemSent,
    abortedLastRun,
    storePath,
    sessionScope,
    groupResolution,
    isGroup,
    triggerBodyNormalized,
    bodyStripped,
  } = sessionState;

  await applyResetModelOverride({
    cfg,
    resetTriggered,
    bodyStripped,
    sessionCtx,
    ctx: finalized,
    sessionEntry,
    sessionStore,
    sessionKey,
    storePath,
    defaultProvider,
    defaultModel,
    aliasIndex,
  });

  // ─────────────────────────────────────────────────────────────────────────────
  // 阶段 5: 模型覆盖解析
  // - 解析 channel 级别的模型覆盖
  // - 若无心跳/会话级覆盖，应用 channel 模型
  // ─────────────────────────────────────────────────────────────────────────────
  const channelModelOverride = resolveChannelModelOverride({
    cfg,
    channel:
      groupResolution?.channel ??
      sessionEntry.channel ??
      sessionEntry.origin?.provider ??
      (typeof finalized.OriginatingChannel === "string"
        ? finalized.OriginatingChannel
        : undefined) ??
      finalized.Provider,
    groupId: groupResolution?.id ?? sessionEntry.groupId,
    groupChannel: sessionEntry.groupChannel ?? sessionCtx.GroupChannel ?? finalized.GroupChannel,
    groupSubject: sessionEntry.subject ?? sessionCtx.GroupSubject ?? finalized.GroupSubject,
    parentSessionKey: sessionCtx.ParentSessionKey,
  });
  const hasSessionModelOverride = Boolean(
    sessionEntry.modelOverride?.trim() || sessionEntry.providerOverride?.trim(),
  );
  if (!hasResolvedHeartbeatModelOverride && !hasSessionModelOverride && channelModelOverride) {
    const resolved = resolveModelRefFromString({
      raw: channelModelOverride.model,
      defaultProvider,
      aliasIndex,
    });
    if (resolved) {
      provider = resolved.ref.provider;
      model = resolved.ref.model;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // 阶段 6: 指令解析
  // - 解析命令、指令、思考级别等
  // - 若指令直接返回回复则提前退出
  // ─────────────────────────────────────────────────────────────────────────────
  const directiveResult = await resolveReplyDirectives({
    ctx: finalized,
    cfg,
    agentId,
    agentDir,
    workspaceDir,
    agentCfg,
    sessionCtx,
    sessionEntry,
    sessionStore,
    sessionKey,
    storePath,
    sessionScope,
    groupResolution,
    isGroup,
    triggerBodyNormalized,
    commandAuthorized,
    defaultProvider,
    defaultModel,
    aliasIndex,
    provider,
    model,
    hasResolvedHeartbeatModelOverride,
    typing,
    opts: resolvedOpts,
    skillFilter: mergedSkillFilter,
  });
  if (directiveResult.kind === "reply") {
    return directiveResult.reply;
  }

  let {
    commandSource,
    command,
    allowTextCommands,
    skillCommands,
    directives,
    cleanedBody,
    elevatedEnabled,
    elevatedAllowed,
    elevatedFailures,
    defaultActivation,
    resolvedThinkLevel,
    resolvedVerboseLevel,
    resolvedReasoningLevel,
    resolvedElevatedLevel,
    execOverrides,
    blockStreamingEnabled,
    blockReplyChunking,
    resolvedBlockStreamingBreak,
    provider: resolvedProvider,
    model: resolvedModel,
    modelState,
    contextTokens,
    inlineStatusRequested,
    directiveAck,
    perMessageQueueMode,
    perMessageQueueOptions,
  } = directiveResult.result;
  provider = resolvedProvider;
  model = resolvedModel;

  // ─────────────────────────────────────────────────────────────────────────────
  // 阶段 7: 内联动作处理
  // - 处理 /reset、/new 等重置命令钩子
  // - 处理内联命令/技能
  // - 若内联动作有回复则返回
  // ─────────────────────────────────────────────────────────────────────────────
  const maybeEmitMissingResetHooks = async () => {
    if (!resetTriggered || !command.isAuthorizedSender || command.resetHookTriggered) {
      return;
    }
    const resetMatch = command.commandBodyNormalized.match(/^\/(new|reset)(?:\s|$)/);
    if (!resetMatch) {
      return;
    }
    const action: ResetCommandAction = resetMatch[1] === "reset" ? "reset" : "new";
    await emitResetCommandHooks({
      action,
      ctx,
      cfg,
      command,
      sessionKey,
      sessionEntry,
      previousSessionEntry,
      workspaceDir,
    });
  };

  const inlineActionResult = await handleInlineActions({
    ctx,
    sessionCtx,
    cfg,
    agentId,
    agentDir,
    sessionEntry,
    previousSessionEntry,
    sessionStore,
    sessionKey,
    storePath,
    sessionScope,
    workspaceDir,
    isGroup,
    opts: resolvedOpts,
    typing,
    allowTextCommands,
    inlineStatusRequested,
    command,
    skillCommands,
    directives,
    cleanedBody,
    elevatedEnabled,
    elevatedAllowed,
    elevatedFailures,
    defaultActivation: () => defaultActivation,
    resolvedThinkLevel,
    resolvedVerboseLevel,
    resolvedReasoningLevel,
    resolvedElevatedLevel,
    resolveDefaultThinkingLevel: modelState.resolveDefaultThinkingLevel,
    provider,
    model,
    contextTokens,
    directiveAck,
    abortedLastRun,
    skillFilter: mergedSkillFilter,
  });
  if (inlineActionResult.kind === "reply") {
    await maybeEmitMissingResetHooks();
    return inlineActionResult.reply;
  }
  await maybeEmitMissingResetHooks();
  directives = inlineActionResult.directives;
  abortedLastRun = inlineActionResult.abortedLastRun ?? abortedLastRun;

  // ─────────────────────────────────────────────────────────────────────────────
  // 阶段 8: 沙箱媒体暂存
  // - 将媒体文件暂存到沙箱工作区供 agent 访问
  // ─────────────────────────────────────────────────────────────────────────────
  await stageSandboxMedia({
    ctx,
    sessionCtx,
    cfg,
    sessionKey,
    workspaceDir,
  });

  // ─────────────────────────────────────────────────────────────────────────────
  // 阶段 9: 执行 AI 回复
  // - 调用 runPreparedReply 运行 AI 模型生成最终回复
  // ─────────────────────────────────────────────────────────────────────────────
  return runPreparedReply({
    ctx,
    sessionCtx,
    cfg,
    agentId,
    agentDir,
    agentCfg,
    sessionCfg,
    commandAuthorized,
    command,
    commandSource,
    allowTextCommands,
    directives,
    defaultActivation,
    resolvedThinkLevel,
    resolvedVerboseLevel,
    resolvedReasoningLevel,
    resolvedElevatedLevel,
    execOverrides,
    elevatedEnabled,
    elevatedAllowed,
    blockStreamingEnabled,
    blockReplyChunking,
    resolvedBlockStreamingBreak,
    modelState,
    provider,
    model,
    perMessageQueueMode,
    perMessageQueueOptions,
    typing,
    opts: resolvedOpts,
    defaultProvider,
    defaultModel,
    timeoutMs,
    isNewSession,
    resetTriggered,
    systemSent,
    sessionEntry,
    sessionStore,
    sessionKey,
    sessionId,
    storePath,
    workspaceDir,
    abortedLastRun,
  });
}
