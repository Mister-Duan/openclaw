import type { TypingCallbacks } from "../../channels/typing.js";
import type { HumanDelayConfig } from "../../config/types.js";
import { sleep } from "../../utils.js";
import type { GetReplyOptions, ReplyPayload } from "../types.js";
import { registerDispatcher } from "./dispatcher-registry.js";
import { normalizeReplyPayload, type NormalizeReplySkipReason } from "./normalize-reply.js";
import type { ResponsePrefixContext } from "./response-prefix-template.js";
import type { TypingController } from "./typing.js";

/** 回复分发类型分类。 */
export type ReplyDispatchKind = "tool" | "block" | "final";

/** 回复分发过程中的投递错误处理器。 */
type ReplyDispatchErrorHandler = (err: unknown, info: { kind: ReplyDispatchKind }) => void;

/** 被跳过的回复处理器（例如空回复或被过滤的回复）。 */
type ReplyDispatchSkipHandler = (
  payload: ReplyPayload,
  info: { kind: ReplyDispatchKind; reason: NormalizeReplySkipReason },
) => void;

/** 执行回复载荷实际投递的异步函数。 */
type ReplyDispatchDeliverer = (
  payload: ReplyPayload,
  info: { kind: ReplyDispatchKind },
) => Promise<void>;

const DEFAULT_HUMAN_DELAY_MIN_MS = 800;
const DEFAULT_HUMAN_DELAY_MAX_MS = 2500;

/**
 * 在配置范围内生成随机的仿人类延迟。
 * 用于在分块回复之间创建自然的节奏，模拟人类打字。
 *
 * @param config - 人类延迟配置，指定模式和可选的最小/最大值。
 * @returns 延迟毫秒数。如果模式为 "off" 则返回 0。
 */
function getHumanDelay(config: HumanDelayConfig | undefined): number {
  const mode = config?.mode ?? "off";
  if (mode === "off") {
    return 0;
  }
  const min =
    mode === "custom" ? (config?.minMs ?? DEFAULT_HUMAN_DELAY_MIN_MS) : DEFAULT_HUMAN_DELAY_MIN_MS;
  const max =
    mode === "custom" ? (config?.maxMs ?? DEFAULT_HUMAN_DELAY_MAX_MS) : DEFAULT_HUMAN_DELAY_MAX_MS;
  if (max <= min) {
    return min;
  }
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * 创建回复分发器的配置选项。
 */
export type ReplyDispatcherOptions = {
  /** 将回复载荷投递到目标频道的函数。 */
  deliver: ReplyDispatchDeliverer;
  /** 可选的响应文本前缀。 */
  responsePrefix?: string;
  /** 响应前缀模板插值的静态上下文。 */
  responsePrefixContext?: ResponsePrefixContext;
  /** 响应前缀模板插值的动态上下文提供器。
   * 在标准化时调用，模型选择完成后。 */
  responsePrefixContextProvider?: () => ResponsePrefixContext;
  /** 当心跳标记从响应中被剥离时的回调。 */
  onHeartbeatStrip?: () => void;
  /** 当分发器变为空闲状态时的回调（无待处理投递）。 */
  onIdle?: () => void;
  /** 投递失败的错误处理器。 */
  onError?: ReplyDispatchErrorHandler;
  // AIDEV-NOTE: onSkip 让频道检测静默/空投递丢弃（例如 Telegram 空响应回退）。
  /** 当回复被跳过时的回调（例如空回复或被过滤）。 */
  onSkip?: ReplyDispatchSkipHandler;
  /** 分块回复之间的仿人类延迟，用于自然节奏。 */
  humanDelay?: HumanDelayConfig;
};

/**
 * 创建带输入指示器支持的回复分发器的扩展选项。
 */
export type ReplyDispatcherWithTypingOptions = Omit<ReplyDispatcherOptions, "onIdle"> & {
  /** 频道的输入指示器回调。 */
  typingCallbacks?: TypingCallbacks;
  /** 回复生成开始时的回调。 */
  onReplyStart?: () => Promise<void> | void;
  /** 分发器变为空闲时的回调。 */
  onIdle?: () => void;
  /** 输入控制器被清理时调用（例如 NO_REPLY 时）。 */
  onCleanup?: () => void;
};

/**
 * 创建带输入支持的回复分发器的返回结果对象。
 */
type ReplyDispatcherWithTypingResult = {
  dispatcher: ReplyDispatcher;
  replyOptions: Pick<GetReplyOptions, "onReplyStart" | "onTypingController" | "onTypingCleanup">;
  markDispatchIdle: () => void;
  /** 通知模型运行完成，以便输入控制器可以停止。 */
  markRunComplete: () => void;
};

/**
 * 回复分发器接口，用于序列化和投递回复载荷。
 * 确保工具结果、分块回复和最终回复按顺序投递。
 */
export type ReplyDispatcher = {
  /** 将工具结果加入投递队列。返回 true 表示载荷被接受。 */
  sendToolResult: (payload: ReplyPayload) => boolean;
  /** 将分块回复加入投递队列。返回 true 表示载荷被接受。 */
  sendBlockReply: (payload: ReplyPayload) => boolean;
  /** 将最终回复加入投递队列。返回 true 表示载荷被接受。 */
  sendFinalReply: (payload: ReplyPayload) => boolean;
  /** 等待所有待处理投递完成。 */
  waitForIdle: () => Promise<void>;
  /** 获取按类型分类的队列中回复数量。 */
  getQueuedCounts: () => Record<ReplyDispatchKind, number>;
  /** 通知不会再有新的回复入队。 */
  markComplete: () => void;
};

/** 标准化回复载荷的内部选项。 */
type NormalizeReplyPayloadInternalOptions = Pick<
  ReplyDispatcherOptions,
  "responsePrefix" | "responsePrefixContext" | "responsePrefixContextProvider" | "onHeartbeatStrip"
> & {
  onSkip?: (reason: NormalizeReplySkipReason) => void;
};

/**
 * 通过应用响应前缀和过滤来标准化回复载荷。
 * 如果同时提供了动态和静态上下文，优先使用动态上下文提供器。
 *
 * @param payload - 要标准化的原始回复载荷。
 * @param opts - 标准化选项，包括前缀上下文和回调。
 * @returns 标准化后的载荷，如果应跳过该载荷则返回 null。
 */
function normalizeReplyPayloadInternal(
  payload: ReplyPayload,
  opts: NormalizeReplyPayloadInternalOptions,
): ReplyPayload | null {
  // 如果同时提供了动态和静态上下文，优先使用动态上下文提供器
  const prefixContext = opts.responsePrefixContextProvider?.() ?? opts.responsePrefixContext;

  return normalizeReplyPayload(payload, {
    responsePrefix: opts.responsePrefix,
    responsePrefixContext: prefixContext,
    onHeartbeatStrip: opts.onHeartbeatStrip,
    onSkip: opts.onSkip,
  });
}

/**
 * 创建一个序列化并投递回复载荷的回复分发器。
 *
 * 分发器维护一个队列以确保回复按顺序投递，
 * 跟踪待处理投递以检测空闲状态，并可选地在分块回复之间
 * 添加仿人类延迟以产生自然的对话节奏。
 *
 * 功能特性：
 * - 序列化出站回复以保持 tool/block/final 顺序
 * - 跟踪待处理计数以协调网关重启
 * - 在分块回复之间添加可配置的仿人类延迟
 * - 使用响应前缀和上下文标准化载荷
 * - 所有投递完成时发出空闲回调
 *
 * @param options - 分发器的配置选项。
 * @returns 包含各类型回复发送方法的 ReplyDispatcher 实例。
 */
export function createReplyDispatcher(options: ReplyDispatcherOptions): ReplyDispatcher {
  let sendChain: Promise<void> = Promise.resolve();
  // 跟踪进行中的投递，以便发出可靠的"空闲"信号。
  // 以 pending=1 作为"预留"开始，防止网关过早重启。
  // 当调用 markComplete() 通知不会再有回复时，此值会递减。
  let pending = 1;
  let completeCalled = false;
  // 跟踪是否已发送分块回复（用于人类延迟 - 跳过第一个分块的延迟）。
  let sentFirstBlock = false;
  // 序列化出站回复以保持 tool/block/final 顺序。
  const queuedCounts: Record<ReplyDispatchKind, number> = {
    tool: 0,
    block: 0,
    final: 0,
  };

  // Register this dispatcher globally for gateway restart coordination.
  const { unregister } = registerDispatcher({
    pending: () => pending,
    waitForIdle: () => sendChain,
  });

  /**
   * 将回复载荷加入队列以按顺序投递。
   * 标准化载荷，跟踪待处理计数，并链式执行投递。
   *
   * @param kind - 回复类型（tool、block 或 final）。
   * @param payload - 要投递的回复载荷。
   * @returns 如果载荷被接受返回 true，被跳过返回 false。
   */
  const enqueue = (kind: ReplyDispatchKind, payload: ReplyPayload) => {
    const normalized = normalizeReplyPayloadInternal(payload, {
      responsePrefix: options.responsePrefix,
      responsePrefixContext: options.responsePrefixContext,
      responsePrefixContextProvider: options.responsePrefixContextProvider,
      onHeartbeatStrip: options.onHeartbeatStrip,
      onSkip: (reason) => options.onSkip?.(payload, { kind, reason }),
    });
    if (!normalized) {
      return false;
    }
    queuedCounts[kind] += 1;
    pending += 1;

    // 判断是否应添加仿人类延迟（仅在第一个分块之后的分块回复时添加）。
    const shouldDelay = kind === "block" && sentFirstBlock;
    if (kind === "block") {
      sentFirstBlock = true;
    }

    sendChain = sendChain
      .then(async () => {
        // 在分块回复之间添加仿人类延迟以产生自然节奏。
        if (shouldDelay) {
          const delayMs = getHumanDelay(options.humanDelay);
          if (delayMs > 0) {
            await sleep(delayMs);
          }
        }
        // 安全：deliver 在异步 .then() 回调内调用，因此即使同步抛出
        // 也会变成通过 .catch()/.finally() 流转的拒绝，确保清理执行。
        await options.deliver(normalized, { kind });
      })
      .catch((err) => {
        options.onError?.(err, { kind });
      })
      .finally(() => {
        pending -= 1;
        // 清除预留条件：
        // 1. pending 现在为 1（只剩预留）
        // 2. markComplete 已被调用
        // 3. 不会再有回复入队
        if (pending === 1 && completeCalled) {
          pending -= 1; // 清除预留
        }
        if (pending === 0) {
          // 空闲时从全局跟踪中注销。
          unregister();
          options.onIdle?.();
        }
      });
    return true;
  };

  /**
   * 通知不会再有新的回复入队。
   * 清除预留计数器，如果没有待处理投递则触发空闲回调。
   * 使用微任务调度以允许任何进行中的 enqueue() 调用先完成。
   */
  const markComplete = () => {
    if (completeCalled) {
      return;
    }
    completeCalled = true;
    // 如果没有回复入队（pending 仍为 1 = 仅预留），
    // 在当前微任务完成后调度清除预留。
    // 这给进行中的 enqueue() 调用一个机会来增加 pending。
    void Promise.resolve().then(() => {
      if (pending === 1 && completeCalled) {
        // 仍然只有预留，没有回复入队
        pending -= 1;
        if (pending === 0) {
          unregister();
          options.onIdle?.();
        }
      }
    });
  };

  return {
    sendToolResult: (payload) => enqueue("tool", payload),
    sendBlockReply: (payload) => enqueue("block", payload),
    sendFinalReply: (payload) => enqueue("final", payload),
    waitForIdle: () => sendChain,
    getQueuedCounts: () => ({ ...queuedCounts }),
    markComplete,
  };
}

/**
 * 创建集成输入指示器支持的回复分发器。
 *
 * 包装 `createReplyDispatcher` 并与输入控制器连接，
 * 在回复生成和投递期间显示/隐藏输入指示器。
 * 当分发器变为空闲或模型运行完成时，输入控制器会被自动通知。
 *
 * @param options - 配置选项，包括输入回调和分发器选项。
 * @returns 包含分发器、模型调用的回复选项，以及手动标记分发空闲或运行完成的函数的对象。
 */
export function createReplyDispatcherWithTyping(
  options: ReplyDispatcherWithTypingOptions,
): ReplyDispatcherWithTypingResult {
  const { typingCallbacks, onReplyStart, onIdle, onCleanup, ...dispatcherOptions } = options;
  const resolvedOnReplyStart = onReplyStart ?? typingCallbacks?.onReplyStart;
  const resolvedOnIdle = onIdle ?? typingCallbacks?.onIdle;
  const resolvedOnCleanup = onCleanup ?? typingCallbacks?.onCleanup;
  let typingController: TypingController | undefined;
  const dispatcher = createReplyDispatcher({
    ...dispatcherOptions,
    onIdle: () => {
      typingController?.markDispatchIdle();
      resolvedOnIdle?.();
    },
  });

  return {
    dispatcher,
    replyOptions: {
      onReplyStart: resolvedOnReplyStart,
      onTypingCleanup: resolvedOnCleanup,
      onTypingController: (typing) => {
        typingController = typing;
      },
    },
    markDispatchIdle: () => {
      typingController?.markDispatchIdle();
      resolvedOnIdle?.();
    },
    markRunComplete: () => {
      typingController?.markRunComplete();
    },
  };
}
