import { resolveSessionAgentId } from "../../agents/agent-scope.js";
import type { OpenClawConfig } from "../../config/config.js";
import { loadSessionStore, resolveStorePath, type SessionEntry } from "../../config/sessions.js";
import { logVerbose } from "../../globals.js";
import { fireAndForgetHook } from "../../hooks/fire-and-forget.js";
import { createInternalHookEvent, triggerInternalHook } from "../../hooks/internal-hooks.js";
import {
  deriveInboundMessageHookContext,
  toInternalMessageReceivedContext,
  toPluginMessageContext,
  toPluginMessageReceivedEvent,
} from "../../hooks/message-hook-mappers.js";
import { isDiagnosticsEnabled } from "../../infra/diagnostic-events.js";
import {
  logMessageProcessed,
  logMessageQueued,
  logSessionStateChange,
} from "../../logging/diagnostic.js";
import { getGlobalHookRunner } from "../../plugins/hook-runner-global.js";
import { resolveSendPolicy } from "../../sessions/send-policy.js";
import { maybeApplyTtsToPayload, normalizeTtsAutoMode, resolveTtsConfig } from "../../tts/tts.js";
import { INTERNAL_MESSAGE_CHANNEL, normalizeMessageChannel } from "../../utils/message-channel.js";
import { getReplyFromConfig } from "../reply.js";
import type { FinalizedMsgContext } from "../templating.js";
import type { GetReplyOptions, ReplyPayload } from "../types.js";
import { formatAbortReplyText, tryFastAbortFromMessage } from "./abort.js";
import { shouldBypassAcpDispatchForCommand, tryDispatchAcpReply } from "./dispatch-acp.js";
import { shouldSkipDuplicateInbound } from "./inbound-dedupe.js";
import type { ReplyDispatcher, ReplyDispatchKind } from "./reply-dispatcher.js";
import { shouldSuppressReasoningPayload } from "./reply-payloads.js";
import { isRoutableChannel, routeReply } from "./route-reply.js";
import { resolveRunTypingPolicy } from "./typing-policy.js";

const AUDIO_PLACEHOLDER_RE = /^<media:audio>(\s*\([^)]*\))?$/i;
const AUDIO_HEADER_RE = /^\[Audio\b/i;
const normalizeMediaType = (value: string): string => value.split(";")[0]?.trim().toLowerCase();

const isInboundAudioContext = (ctx: FinalizedMsgContext): boolean => {
  const rawTypes = [
    typeof ctx.MediaType === "string" ? ctx.MediaType : undefined,
    ...(Array.isArray(ctx.MediaTypes) ? ctx.MediaTypes : []),
  ].filter(Boolean) as string[];
  const types = rawTypes.map((type) => normalizeMediaType(type));
  if (types.some((type) => type === "audio" || type.startsWith("audio/"))) {
    return true;
  }

  const body =
    typeof ctx.BodyForCommands === "string"
      ? ctx.BodyForCommands
      : typeof ctx.CommandBody === "string"
        ? ctx.CommandBody
        : typeof ctx.RawBody === "string"
          ? ctx.RawBody
          : typeof ctx.Body === "string"
            ? ctx.Body
            : "";
  const trimmed = body.trim();
  if (!trimmed) {
    return false;
  }
  if (AUDIO_PLACEHOLDER_RE.test(trimmed)) {
    return true;
  }
  return AUDIO_HEADER_RE.test(trimmed);
};

const resolveSessionStoreEntry = (
  ctx: FinalizedMsgContext,
  cfg: OpenClawConfig,
): {
  sessionKey?: string;
  entry?: SessionEntry;
} => {
  const targetSessionKey =
    ctx.CommandSource === "native" ? ctx.CommandTargetSessionKey?.trim() : undefined;
  const sessionKey = (targetSessionKey ?? ctx.SessionKey)?.trim();
  if (!sessionKey) {
    return {};
  }
  const agentId = resolveSessionAgentId({ sessionKey, config: cfg });
  const storePath = resolveStorePath(cfg.session?.store, { agentId });
  try {
    const store = loadSessionStore(storePath);
    return {
      sessionKey,
      entry: store[sessionKey.toLowerCase()] ?? store[sessionKey],
    };
  } catch {
    return {
      sessionKey,
    };
  }
};

export type DispatchFromConfigResult = {
  queuedFinal: boolean;
  counts: Record<ReplyDispatchKind, number>;
};

/**
 * 根据配置分发回复消息。
 *
 * 这是消息处理的核心函数，负责协调整个回复生成和分发流程。
 *
 * ## 主要功能：
 * 1. **消息去重**：检测并跳过重复的入站消息
 * 2. **钩子触发**：触发插件钩子和内部钩子（message_received）
 * 3. **跨频道路由**：支持将回复路由到不同的消息频道
 * 4. **快速中止**：检测中止命令并立即响应
 * 5. **发送策略**：根据配置的发送策略决定是否发送消息
 * 6. **ACP 分发**：尝试通过 ACP（Agent Communication Protocol）分发回复
 * 7. **流式回复**：支持工具结果、分块回复和最终回复的流式处理
 * 8. **TTS 处理**：为回复添加文字转语音音频
 * 9. **诊断日志**：记录消息处理的各个阶段
 *
 * ## 处理流程：
 * ```
 * 入站消息 → 去重检查 → 触发钩子 → 路由决策 → 快速中止检查
 *     ↓
 * 发送策略检查 → ACP 分发尝试 → 获取回复 → TTS 处理 → 分发回复
 * ```
 *
 * @param params - 分发参数
 * @param params.ctx - 已格式化的消息上下文，包含发送者、接收者、消息内容等
 * @param params.cfg - OpenClaw 配置对象
 * @param params.dispatcher - 回复分发器，负责序列化和投递回复
 * @param params.replyOptions - 可选的回复生成选项
 * @param params.replyResolver - 可选的回复解析器，默认使用 getReplyFromConfig
 *
 * @returns 分发结果，包含是否成功入队最终回复以及各类型回复的计数
 */
export async function dispatchReplyFromConfig(params: {
  ctx: FinalizedMsgContext;
  cfg: OpenClawConfig;
  dispatcher: ReplyDispatcher;
  replyOptions?: Omit<GetReplyOptions, "onToolResult" | "onBlockReply">;
  replyResolver?: typeof getReplyFromConfig;
}): Promise<DispatchFromConfigResult> {
  const { ctx, cfg, dispatcher } = params;
  // ==================== 初始化诊断和跟踪变量 ====================
  const diagnosticsEnabled = isDiagnosticsEnabled(cfg);
  const channel = String(ctx.Surface ?? ctx.Provider ?? "unknown").toLowerCase();
  const chatId = ctx.To ?? ctx.From;
  const messageId = ctx.MessageSid ?? ctx.MessageSidFirst ?? ctx.MessageSidLast;
  const sessionKey = ctx.SessionKey;
  const startTime = diagnosticsEnabled ? Date.now() : 0;
  const canTrackSession = diagnosticsEnabled && Boolean(sessionKey);

  /**
   * 记录消息处理结果（用于诊断日志）
   * @param outcome - 处理结果：completed（完成）、skipped（跳过）、error（错误）
   * @param opts - 可选的原因和错误信息
   */
  const recordProcessed = (
    outcome: "completed" | "skipped" | "error",
    opts?: {
      reason?: string;
      error?: string;
    },
  ) => {
    if (!diagnosticsEnabled) {
      return;
    }
    logMessageProcessed({
      channel,
      chatId,
      messageId,
      sessionKey,
      durationMs: Date.now() - startTime,
      outcome,
      reason: opts?.reason,
      error: opts?.error,
    });
  };

  /** 标记会话进入处理状态 */
  const markProcessing = () => {
    if (!canTrackSession || !sessionKey) {
      return;
    }
    logMessageQueued({ sessionKey, channel, source: "dispatch" });
    logSessionStateChange({
      sessionKey,
      state: "processing",
      reason: "message_start",
    });
  };

  /** 标记会话进入空闲状态 */
  const markIdle = (reason: string) => {
    if (!canTrackSession || !sessionKey) {
      return;
    }
    logSessionStateChange({
      sessionKey,
      state: "idle",
      reason,
    });
  };

  // ==================== 重复消息检查 ====================
  // 跳过重复的入站消息以避免重复处理
  if (shouldSkipDuplicateInbound(ctx)) {
    recordProcessed("skipped", { reason: "duplicate" });
    return { queuedFinal: false, counts: dispatcher.getQueuedCounts() };
  }

  // ==================== 解析会话状态 ====================
  // 从会话存储中获取会话条目（包含 TTS 模式、频道信息等）
  const sessionStoreEntry = resolveSessionStoreEntry(ctx, cfg);
  const acpDispatchSessionKey = sessionStoreEntry.sessionKey ?? sessionKey;
  const inboundAudio = isInboundAudioContext(ctx); // 检测入站消息是否为音频
  const sessionTtsAuto = normalizeTtsAutoMode(sessionStoreEntry.entry?.ttsAuto);
  const hookRunner = getGlobalHookRunner();

  // ==================== 提取钩子上下文 ====================
  // Extract message context for hooks (plugin and internal)
  const timestamp =
    typeof ctx.Timestamp === "number" && Number.isFinite(ctx.Timestamp) ? ctx.Timestamp : undefined;
  const messageIdForHook =
    ctx.MessageSidFull ?? ctx.MessageSid ?? ctx.MessageSidFirst ?? ctx.MessageSidLast;
  const hookContext = deriveInboundMessageHookContext(ctx, { messageId: messageIdForHook });
  const { isGroup, groupId } = hookContext;

  // ==================== 触发插件钩子（异步，不等待结果） ====================
  // Trigger plugin hooks (fire-and-forget)
  if (hookRunner?.hasHooks("message_received")) {
    fireAndForgetHook(
      hookRunner.runMessageReceived(
        toPluginMessageReceivedEvent(hookContext),
        toPluginMessageContext(hookContext),
      ),
      "dispatch-from-config: message_received plugin hook failed",
    );
  }

  // ==================== 触发内部钩子（HOOK.md 发现系统） ====================
  // Bridge to internal hooks (HOOK.md discovery system) - refs #8807
  if (sessionKey) {
    fireAndForgetHook(
      triggerInternalHook(
        createInternalHookEvent("message", "received", sessionKey, {
          ...toInternalMessageReceivedContext(hookContext),
          timestamp,
        }),
      ),
      "dispatch-from-config: message_received internal hook failed",
    );
  }

  // ==================== 跨频道路由决策 ====================
  // Check if we should route replies to originating channel instead of dispatcher.
  // 检查是否应将回复路由到原始频道而非分发器。
  // Only route when the originating channel is DIFFERENT from the current surface.
  // 仅当原始频道与当前 surface 不同时才路由。
  // This handles cross-provider routing (e.g., message from Telegram being processed
  // 这处理跨提供商路由（例如，来自 Telegram 的消息被共享会话处理，
  // by a shared session that's currently on Slack) while preserving normal dispatcher
  // 该会话当前在 Slack 上），同时保留提供商处理自己消息时的正常分发器流程。
  // flow when the provider handles its own messages.
  //
  // Debug: `pnpm test src/auto-reply/reply/dispatch-from-config.test.ts`
  const originatingChannel = normalizeMessageChannel(ctx.OriginatingChannel);
  const originatingTo = ctx.OriginatingTo;
  const providerChannel = normalizeMessageChannel(ctx.Provider);
  const surfaceChannel = normalizeMessageChannel(ctx.Surface);
  // Prefer provider channel because surface may carry origin metadata in relayed flows.
  // 优先使用 provider 频道，因为在中继流程中 surface 可能携带原始元数据。
  const currentSurface = providerChannel ?? surfaceChannel;
  const shouldRouteToOriginating = Boolean(
    isRoutableChannel(originatingChannel) && originatingTo && originatingChannel !== currentSurface,
  );
  // 跨频道路由或内部消息时抑制输入指示器
  const shouldSuppressTyping =
    shouldRouteToOriginating || originatingChannel === INTERNAL_MESSAGE_CHANNEL;
  const ttsChannel = shouldRouteToOriginating ? originatingChannel : currentSurface;

  /**
   * 异步发送载荷到路由频道的辅助函数。
   * 仅在实际路由到不同提供商时使用。
   * 注意：仅在 shouldRouteToOriginating 为 true 时调用，
   * 所以 originatingChannel 和 originatingTo 保证已定义。
   */
  const sendPayloadAsync = async (
    payload: ReplyPayload,
    abortSignal?: AbortSignal,
    mirror?: boolean,
  ): Promise<void> => {
    // TypeScript 无法从 shouldRouteToOriginating 检查中收窄这些类型，
    // 但当此函数被调用时它们保证非空。
    // TypeScript doesn't narrow these from the shouldRouteToOriginating check,
    // but they're guaranteed non-null when this function is called.
    if (!originatingChannel || !originatingTo) {
      return;
    }
    if (abortSignal?.aborted) {
      return;
    }
    const result = await routeReply({
      payload,
      channel: originatingChannel,
      to: originatingTo,
      sessionKey: ctx.SessionKey,
      accountId: ctx.AccountId,
      threadId: ctx.MessageThreadId,
      cfg,
      abortSignal,
      mirror,
      isGroup,
      groupId,
    });
    if (!result.ok) {
      logVerbose(`dispatch-from-config: route-reply failed: ${result.error ?? "unknown error"}`);
    }
  };

  // 标记会话开始处理
  markProcessing();

  try {
    // ==================== 快速中止检查 ====================
    // 检测用户的中止命令（如 /stop），立即停止正在进行的生成
    const fastAbort = await tryFastAbortFromMessage({ ctx, cfg });
    if (fastAbort.handled) {
      const payload = {
        text: formatAbortReplyText(fastAbort.stoppedSubagents),
      } satisfies ReplyPayload;
      let queuedFinal = false;
      let routedFinalCount = 0;
      if (shouldRouteToOriginating && originatingChannel && originatingTo) {
        const result = await routeReply({
          payload,
          channel: originatingChannel,
          to: originatingTo,
          sessionKey: ctx.SessionKey,
          accountId: ctx.AccountId,
          threadId: ctx.MessageThreadId,
          cfg,
          isGroup,
          groupId,
        });
        queuedFinal = result.ok;
        if (result.ok) {
          routedFinalCount += 1;
        }
        if (!result.ok) {
          logVerbose(
            `dispatch-from-config: route-reply (abort) failed: ${result.error ?? "unknown error"}`,
          );
        }
      } else {
        queuedFinal = dispatcher.sendFinalReply(payload);
      }
      const counts = dispatcher.getQueuedCounts();
      counts.final += routedFinalCount;
      recordProcessed("completed", { reason: "fast_abort" });
      markIdle("message_completed");
      return { queuedFinal, counts };
    }

    // 检查是否应为命令绕过 ACP 分发
    const bypassAcpForCommand = shouldBypassAcpDispatchForCommand(ctx, cfg);

    // ==================== 发送策略检查 ====================
    // 根据配置的发送策略决定是否允许发送消息
    const sendPolicy = resolveSendPolicy({
      cfg,
      entry: sessionStoreEntry.entry,
      sessionKey: sessionStoreEntry.sessionKey ?? sessionKey,
      channel:
        sessionStoreEntry.entry?.channel ??
        ctx.OriginatingChannel ??
        ctx.Surface ??
        ctx.Provider ??
        undefined,
      chatType: sessionStoreEntry.entry?.chatType,
    });
    if (sendPolicy === "deny" && !bypassAcpForCommand) {
      logVerbose(
        `Send blocked by policy for session ${sessionStoreEntry.sessionKey ?? sessionKey ?? "unknown"}`,
      );
      const counts = dispatcher.getQueuedCounts();
      recordProcessed("completed", { reason: "send_policy_deny" });
      markIdle("message_completed");
      return { queuedFinal: false, counts };
    }

    // 是否发送工具摘要（群组和原生命令不发送）
    const shouldSendToolSummaries = ctx.ChatType !== "group" && ctx.CommandSource !== "native";

    // ==================== ACP 分发尝试 ====================
    // 尝试通过 ACP（Agent Communication Protocol）分发回复
    const acpDispatch = await tryDispatchAcpReply({
      ctx,
      cfg,
      dispatcher,
      sessionKey: acpDispatchSessionKey,
      inboundAudio,
      sessionTtsAuto,
      ttsChannel,
      shouldRouteToOriginating,
      originatingChannel,
      originatingTo,
      shouldSendToolSummaries,
      bypassForCommand: bypassAcpForCommand,
      onReplyStart: params.replyOptions?.onReplyStart,
      recordProcessed,
      markIdle,
    });
    if (acpDispatch) {
      return acpDispatch;
    }

    // ==================== 分块回复 TTS 累积 ====================
    // Track accumulated block text for TTS generation after streaming completes.
    // 跟踪累积的分块文本，用于流式传输完成后生成 TTS。
    // When block streaming succeeds, there's no final reply, so we need to generate
    // 当分块流式传输成功时，没有最终回复，因此需要从累积的分块内容
    // TTS audio separately from the accumulated block content.
    // 单独生成 TTS 音频。
    let accumulatedBlockText = "";
    let blockCount = 0;

    /**
     * 解析工具投递载荷。
     * 群组/原生流程故意抑制工具摘要文本，但仅包含媒体的工具结果（如 TTS 音频）仍需投递。
     */
    const resolveToolDeliveryPayload = (payload: ReplyPayload): ReplyPayload | null => {
      if (shouldSendToolSummaries) {
        return payload;
      }
      // Group/native flows intentionally suppress tool summary text, but media-only
      // tool results (for example TTS audio) must still be delivered.
      const hasMedia = Boolean(payload.mediaUrl) || (payload.mediaUrls?.length ?? 0) > 0;
      if (!hasMedia) {
        return null;
      }
      return { ...payload, text: undefined };
    };
    // 解析输入指示器策略
    const typing = resolveRunTypingPolicy({
      requestedPolicy: params.replyOptions?.typingPolicy,
      suppressTyping: params.replyOptions?.suppressTyping === true || shouldSuppressTyping,
      originatingChannel,
      systemEvent: shouldRouteToOriginating,
    });

    // ==================== 获取回复并处理流式输出 ====================
    const replyResult = await (params.replyResolver ?? getReplyFromConfig)(
      ctx,
      {
        ...params.replyOptions,
        typingPolicy: typing.typingPolicy,
        suppressTyping: typing.suppressTyping,
        // 工具结果回调：处理工具执行结果并投递
        onToolResult: (payload: ReplyPayload) => {
          const run = async () => {
            const ttsPayload = await maybeApplyTtsToPayload({
              payload,
              cfg,
              channel: ttsChannel,
              kind: "tool",
              inboundAudio,
              ttsAuto: sessionTtsAuto,
            });
            const deliveryPayload = resolveToolDeliveryPayload(ttsPayload);
            if (!deliveryPayload) {
              return;
            }
            if (shouldRouteToOriginating) {
              await sendPayloadAsync(deliveryPayload, undefined, false);
            } else {
              dispatcher.sendToolResult(deliveryPayload);
            }
          };
          return run();
        },
        // 分块回复回调：处理流式输出的每个分块
        onBlockReply: (payload: ReplyPayload, context) => {
          const run = async () => {
            // 抑制推理载荷 — 使用此通用分发路径的频道
            // （WhatsApp、web 等）没有专用的推理通道。
            // Telegram 有自己的分发路径来处理推理分割。
            // Suppress reasoning payloads — channels using this generic dispatch
            // path (WhatsApp, web, etc.) do not have a dedicated reasoning lane.
            // Telegram has its own dispatch path that handles reasoning splitting.
            if (shouldSuppressReasoningPayload(payload)) {
              return;
            }
            // 累积分块文本以便在流式传输后生成 TTS
            // Accumulate block text for TTS generation after streaming
            if (payload.text) {
              if (accumulatedBlockText.length > 0) {
                accumulatedBlockText += "\n";
              }
              accumulatedBlockText += payload.text;
              blockCount++;
            }
            const ttsPayload = await maybeApplyTtsToPayload({
              payload,
              cfg,
              channel: ttsChannel,
              kind: "block",
              inboundAudio,
              ttsAuto: sessionTtsAuto,
            });
            if (shouldRouteToOriginating) {
              await sendPayloadAsync(ttsPayload, context?.abortSignal, false);
            } else {
              dispatcher.sendBlockReply(ttsPayload);
            }
          };
          return run();
        },
      },
      cfg,
    );

    // ==================== ACP 尾部分发（重置后） ====================
    if (ctx.AcpDispatchTailAfterReset === true) {
      // 命令处理准备了 ACP 就地重置后的尾部提示。
      // 现在通过 ACP 路由该尾部（同一回合），而不是嵌入式分发。
      // Command handling prepared a trailing prompt after ACP in-place reset.
      // Route that tail through ACP now (same turn) instead of embedded dispatch.
      ctx.AcpDispatchTailAfterReset = false;
      const acpTailDispatch = await tryDispatchAcpReply({
        ctx,
        cfg,
        dispatcher,
        sessionKey: acpDispatchSessionKey,
        inboundAudio,
        sessionTtsAuto,
        ttsChannel,
        shouldRouteToOriginating,
        originatingChannel,
        originatingTo,
        shouldSendToolSummaries,
        bypassForCommand: false,
        onReplyStart: params.replyOptions?.onReplyStart,
        recordProcessed,
        markIdle,
      });
      if (acpTailDispatch) {
        return acpTailDispatch;
      }
    }

    // ==================== 处理最终回复 ====================
    const replies = replyResult ? (Array.isArray(replyResult) ? replyResult : [replyResult]) : [];

    let queuedFinal = false;
    let routedFinalCount = 0;
    for (const reply of replies) {
      // 从频道投递中抑制推理载荷 — 使用此通用分发路径的频道没有专用的推理通道。
      // Suppress reasoning payloads from channel delivery — channels using this
      // generic dispatch path do not have a dedicated reasoning lane.
      if (shouldSuppressReasoningPayload(reply)) {
        continue;
      }
      // 为回复添加 TTS 音频
      const ttsReply = await maybeApplyTtsToPayload({
        payload: reply,
        cfg,
        channel: ttsChannel,
        kind: "final",
        inboundAudio,
        ttsAuto: sessionTtsAuto,
      });
      if (shouldRouteToOriginating && originatingChannel && originatingTo) {
        // 将最终回复路由到原始频道。
        // Route final reply to originating channel.
        const result = await routeReply({
          payload: ttsReply,
          channel: originatingChannel,
          to: originatingTo,
          sessionKey: ctx.SessionKey,
          accountId: ctx.AccountId,
          threadId: ctx.MessageThreadId,
          cfg,
          isGroup,
          groupId,
        });
        if (!result.ok) {
          logVerbose(
            `dispatch-from-config: route-reply (final) failed: ${result.error ?? "unknown error"}`,
          );
        }
        queuedFinal = result.ok || queuedFinal;
        if (result.ok) {
          routedFinalCount += 1;
        }
      } else {
        queuedFinal = dispatcher.sendFinalReply(ttsReply) || queuedFinal;
      }
    }

    const ttsMode = resolveTtsConfig(cfg).mode ?? "final";
    // ==================== 累积分块的 TTS 生成 ====================
    // Generate TTS-only reply after block streaming completes (when there's no final reply).
    // 分块流式传输完成后生成仅 TTS 回复（当没有最终回复时）。
    // This handles the case where block streaming succeeds and drops final payloads,
    // 这处理分块流式传输成功并丢弃最终载荷的情况，
    // but we still want TTS audio to be generated from the accumulated block content.
    // 但我们仍然希望从累积的分块内容生成 TTS 音频。
    if (
      ttsMode === "final" &&
      replies.length === 0 &&
      blockCount > 0 &&
      accumulatedBlockText.trim()
    ) {
      try {
        const ttsSyntheticReply = await maybeApplyTtsToPayload({
          payload: { text: accumulatedBlockText },
          cfg,
          channel: ttsChannel,
          kind: "final",
          inboundAudio,
          ttsAuto: sessionTtsAuto,
        });
        // Only send if TTS was actually applied (mediaUrl exists)
        if (ttsSyntheticReply.mediaUrl) {
          // Send TTS-only payload (no text, just audio) so it doesn't duplicate the block content
          const ttsOnlyPayload: ReplyPayload = {
            mediaUrl: ttsSyntheticReply.mediaUrl,
            audioAsVoice: ttsSyntheticReply.audioAsVoice,
          };
          if (shouldRouteToOriginating && originatingChannel && originatingTo) {
            const result = await routeReply({
              payload: ttsOnlyPayload,
              channel: originatingChannel,
              to: originatingTo,
              sessionKey: ctx.SessionKey,
              accountId: ctx.AccountId,
              threadId: ctx.MessageThreadId,
              cfg,
              isGroup,
              groupId,
            });
            queuedFinal = result.ok || queuedFinal;
            if (result.ok) {
              routedFinalCount += 1;
            }
            if (!result.ok) {
              logVerbose(
                `dispatch-from-config: route-reply (tts-only) failed: ${result.error ?? "unknown error"}`,
              );
            }
          } else {
            const didQueue = dispatcher.sendFinalReply(ttsOnlyPayload);
            queuedFinal = didQueue || queuedFinal;
          }
        }
      } catch (err) {
        logVerbose(
          `dispatch-from-config: accumulated block TTS failed: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    }

    const counts = dispatcher.getQueuedCounts();
    counts.final += routedFinalCount;
    recordProcessed("completed");
    markIdle("message_completed");
    return { queuedFinal, counts };
  } catch (err) {
    recordProcessed("error", { error: String(err) });
    markIdle("message_error");
    throw err;
  }
}
