"use client";

import type { UseChatHelpers } from "@ai-sdk/react";
import { useEffect, useRef } from "react";
import { useDataStream } from "@/components/data-stream-provider";
import type { ChatMessage } from "@/lib/types";

export type UseAutoResumeParams = {
  autoResume: boolean;
  initialMessages: ChatMessage[];
  resumeStream: UseChatHelpers<ChatMessage>["resumeStream"];
  setMessages: UseChatHelpers<ChatMessage>["setMessages"];
  status?: UseChatHelpers<ChatMessage>["status"];
};

export function useAutoResume({
  autoResume,
  initialMessages,
  resumeStream,
  setMessages,
  status,
}: UseAutoResumeParams) {
  const { dataStream } = useDataStream();
  const hasResumedRef = useRef(false);

  useEffect(() => {
    if (!autoResume) {
      return;
    }

    // Prevent multiple resume attempts
    if (hasResumedRef.current) {
      return;
    }

    // Only try to resume when status is ready (chat is initialized)
    if (status && status !== "ready") {
      return;
    }

    const mostRecentMessage = initialMessages.at(-1);

    if (mostRecentMessage?.role === "user") {
      hasResumedRef.current = true;
      // Wrap in try-catch to handle cases where internal state isn't ready
      try {
        resumeStream();
      } catch (error) {
        console.warn("[useAutoResume] Failed to resume stream:", error);
        hasResumedRef.current = false; // Reset on error to allow retry
      }
    }

    // we intentionally run this once
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoResume, initialMessages.at, resumeStream, status]);

  useEffect(() => {
    if (!dataStream) {
      return;
    }
    if (dataStream.length === 0) {
      return;
    }

    const dataPart = dataStream[0];

    if (dataPart.type === "data-appendMessage") {
      const message = JSON.parse(dataPart.data);
      setMessages([...initialMessages, message]);
    }
  }, [dataStream, initialMessages, setMessages]);
}
