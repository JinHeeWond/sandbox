"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import useSWR, { useSWRConfig } from "swr";
import { ChatHeader } from "@/components/chat-header";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useArtifactSelector } from "@/hooks/use-artifact";
import { useAutoResume } from "@/hooks/use-auto-resume";
import { useChatVisibility } from "@/hooks/use-chat-visibility";
import type { Vote } from "@/lib/db/schema";
import { ChatSDKError } from "@/lib/errors";
import type { Attachment, ChatMessage } from "@/lib/types";
import { fetcher, fetchWithErrorHandlers, generateUUID } from "@/lib/utils";
import { Artifact } from "./artifact";
import { useDataStream } from "./data-stream-provider";
import { Messages } from "./messages";
import { MultimodalInput } from "./multimodal-input";
import { toast } from "./toast";
import type { VisibilityType } from "./visibility-selector";

export function Chat({
  id,
  initialMessages,
  initialChatModel,
  initialVisibilityType,
  isReadonly,
  autoResume,
}: {
  id: string;
  initialMessages: ChatMessage[];
  initialChatModel: string;
  initialVisibilityType: VisibilityType;
  isReadonly: boolean;
  autoResume: boolean;
}) {
  const router = useRouter();

  const { visibilityType } = useChatVisibility({
    chatId: id,
    initialVisibilityType,
  });

  const { mutate } = useSWRConfig();

  // Handle browser back/forward navigation
  useEffect(() => {
    const handlePopState = () => {
      // When user navigates back/forward, refresh to sync with URL
      router.refresh();
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, [router]);
  const { setDataStream } = useDataStream();

  const [input, setInput] = useState<string>("");
  const [showCreditCardAlert, setShowCreditCardAlert] = useState(false);
  const [currentModelId, setCurrentModelId] = useState(initialChatModel);
  const currentModelIdRef = useRef(currentModelId);

  // Ref to store clarifying questions from onData to use in onFinish
  const clarifyingQuestionsRef = useRef<any[] | null>(null);

  useEffect(() => {
    currentModelIdRef.current = currentModelId;
  }, [currentModelId]);

  const {
    messages,
    setMessages,
    sendMessage,
    status,
    stop,
    regenerate,
    resumeStream,
    addToolApprovalResponse,
  } = useChat<ChatMessage>({
    id,
    messages: initialMessages,
    generateId: generateUUID,
    sendAutomaticallyWhen: ({ messages: currentMessages }) => {
      const lastMessage = currentMessages.at(-1);
      const shouldContinue =
        lastMessage?.parts?.some(
          (part) =>
            "state" in part &&
            part.state === "approval-responded" &&
            "approval" in part &&
            (part.approval as { approved?: boolean })?.approved === true
        ) ?? false;
      return shouldContinue;
    },
    transport: new DefaultChatTransport({
      api: "/api/chat",
      fetch: fetchWithErrorHandlers,
      prepareSendMessagesRequest(request) {
        const lastMessage = request.messages.at(-1);
        const isToolApprovalContinuation =
          lastMessage?.role !== "user" ||
          request.messages.some((msg) =>
            msg.parts?.some((part) => {
              const state = (part as { state?: string }).state;
              return (
                state === "approval-responded" || state === "output-denied"
              );
            })
          );

        return {
          body: {
            id: request.id,
            ...(isToolApprovalContinuation
              ? { messages: request.messages }
              : { message: lastMessage }),
            selectedChatModel: currentModelIdRef.current,
            selectedVisibilityType: visibilityType,
            ...request.body,
          },
        };
      },
    }),
    onData: (dataPart) => {
      setDataStream((ds) => (ds ? [...ds, dataPart] : []));

      // Handle clarifying questions data - add to message immediately
      const data = dataPart as any;
      let questions: any[] | null = null;

      // Debug: log all data parts to see the format
      console.log("[onData] Received data part:", JSON.stringify(data, null, 2));

      // Case 1: Direct clarifying-questions data (each item from data array comes individually)
      if (data.type === "clarifying-questions" && data.questions) {
        console.log("[onData] Case 1: Direct clarifying-questions");
        questions = data.questions;
      }
      // Case 2: Wrapped in data array (fallback)
      else if (data.type === "data" && Array.isArray(data.data)) {
        console.log("[onData] Case 2: Wrapped in data array");
        const clarifyingQuestionsData = data.data.find(
          (item: any) => item.type === "clarifying-questions"
        );
        if (clarifyingQuestionsData?.questions) {
          questions = clarifyingQuestionsData.questions;
        }
      }
      // Case 3: Array containing clarifying-questions object
      else if (Array.isArray(data)) {
        console.log("[onData] Case 3: Array data");
        const clarifyingQuestionsData = data.find(
          (item: any) => item.type === "clarifying-questions"
        );
        if (clarifyingQuestionsData?.questions) {
          questions = clarifyingQuestionsData.questions;
        }
      }

      // Immediately add clarifying questions to the last assistant message
      if (questions) {
        console.log("[onData] Found questions, adding to message:", questions);
        clarifyingQuestionsRef.current = questions;
        setMessages((currentMessages) => {
          const lastMessage = currentMessages.at(-1);
          console.log("[onData] Last message role:", lastMessage?.role);
          if (lastMessage?.role === "assistant") {
            // Check if clarifying-questions part already exists
            const hasQuestions = lastMessage.parts?.some(
              (p) => (p as any).type === "clarifying-questions"
            );
            if (!hasQuestions) {
              console.log("[onData] Adding clarifying-questions part to message");
              return currentMessages.map((msg, idx) =>
                idx === currentMessages.length - 1
                  ? {
                      ...msg,
                      parts: [
                        ...(msg.parts || []),
                        {
                          type: "clarifying-questions",
                          questions: questions,
                        } as any,
                      ],
                    }
                  : msg
              );
            }
          }
          return currentMessages;
        });
      }
    },
    onFinish: async () => {
      console.log("[onFinish] Stream finished, clarifyingQuestionsRef:", clarifyingQuestionsRef.current);

      // Clarifying questions가 onData에서 처리되지 않았으면 여기서 처리
      if (clarifyingQuestionsRef.current) {
        const questions = clarifyingQuestionsRef.current;
        clarifyingQuestionsRef.current = null;

        console.log("[onFinish] Processing clarifying questions in onFinish");
        setMessages((currentMessages) => {
          const lastMessage = currentMessages.at(-1);
          console.log("[onFinish] Last message:", lastMessage?.role, lastMessage?.parts?.length);
          if (lastMessage?.role === "assistant") {
            const hasQuestions = lastMessage.parts?.some(
              (p) => (p as any).type === "clarifying-questions"
            );
            if (!hasQuestions) {
              console.log("[onFinish] Adding clarifying-questions to message");
              return currentMessages.map((msg, idx) =>
                idx === currentMessages.length - 1
                  ? {
                      ...msg,
                      parts: [
                        ...(msg.parts || []),
                        {
                          type: "clarifying-questions",
                          questions: questions,
                        } as any,
                      ],
                    }
                  : msg
              );
            } else {
              console.log("[onFinish] Message already has clarifying-questions");
            }
          }
          return currentMessages;
        });
      } else {
        // onData에서 clarifying questions를 못 받았으면 DB에서 최신 메시지를 가져와서 확인
        console.log("[onFinish] No clarifying questions in ref, fetching from DB...");
        try {
          const response = await fetch(`/api/chat/${id}/messages`);
          if (response.ok) {
            const dbMessages = await response.json();
            const lastDbMessage = dbMessages.at(-1);
            if (lastDbMessage?.role === "assistant") {
              const clarifyingPart = lastDbMessage.parts?.find(
                (p: any) => p.type === "clarifying-questions"
              );
              if (clarifyingPart?.questions) {
                console.log("[onFinish] Found clarifying questions in DB, updating messages");
                setMessages((currentMessages) => {
                  const lastMessage = currentMessages.at(-1);
                  if (lastMessage?.role === "assistant") {
                    const hasQuestions = lastMessage.parts?.some(
                      (p) => (p as any).type === "clarifying-questions"
                    );
                    if (!hasQuestions) {
                      return currentMessages.map((msg, idx) =>
                        idx === currentMessages.length - 1
                          ? {
                              ...msg,
                              parts: [
                                ...(msg.parts || []),
                                clarifyingPart as any,
                              ],
                            }
                          : msg
                      );
                    }
                  }
                  return currentMessages;
                });
              }
            }
          }
        } catch (error) {
          console.error("[onFinish] Error fetching messages from DB:", error);
        }
      }

      // 사이드바 히스토리 업데이트를 위해 revalidation 강제
      mutate(
        (key) => typeof key === 'string' && key.startsWith('/api/history'),
        undefined,
        { revalidate: true }
      );
    },
    onError: (error) => {
      if (error instanceof ChatSDKError) {
        if (
          error.message?.includes("AI Gateway requires a valid credit card")
        ) {
          setShowCreditCardAlert(true);
        } else {
          toast({
            type: "error",
            description: error.message,
          });
        }
      }
    },
  });

  const searchParams = useSearchParams();
  const query = searchParams.get("query");

  const [hasAppendedQuery, setHasAppendedQuery] = useState(false);

  useEffect(() => {
    if (query && !hasAppendedQuery) {
      sendMessage({
        role: "user" as const,
        parts: [{ type: "text", text: query }],
      });

      setHasAppendedQuery(true);
      window.history.replaceState({}, "", `/chat/${id}`);
    }
  }, [query, sendMessage, hasAppendedQuery, id]);

  const { data: votes } = useSWR<Vote[]>(
    messages.length >= 2 ? `/api/vote?chatId=${id}` : null,
    fetcher
  );

  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const isArtifactVisible = useArtifactSelector((state) => state.isVisible);

  useAutoResume({
    autoResume,
    initialMessages,
    resumeStream,
    setMessages,
  });

  return (
    <>
      <div className="overscroll-behavior-contain flex h-dvh min-w-0 touch-pan-y flex-col bg-background">
        <ChatHeader
          chatId={id}
          isReadonly={isReadonly}
          selectedVisibilityType={initialVisibilityType}
        />

        <Messages
          addToolApprovalResponse={addToolApprovalResponse}
          chatId={id}
          isArtifactVisible={isArtifactVisible}
          isReadonly={isReadonly}
          messages={messages}
          regenerate={regenerate}
          selectedModelId={initialChatModel}
          sendMessage={sendMessage}
          setMessages={setMessages}
          status={status}
          votes={votes}
        />

        <div className="sticky bottom-0 z-1 mx-auto flex w-full max-w-4xl gap-2 border-t-0 bg-background px-2 pb-3 md:px-4 md:pb-4">
          {!isReadonly && (
            <MultimodalInput
              attachments={attachments}
              chatId={id}
              input={input}
              messages={messages}
              onModelChange={setCurrentModelId}
              selectedModelId={currentModelId}
              selectedVisibilityType={visibilityType}
              sendMessage={sendMessage}
              setAttachments={setAttachments}
              setInput={setInput}
              setMessages={setMessages}
              status={status}
              stop={stop}
            />
          )}
        </div>
      </div>

      <Artifact
        addToolApprovalResponse={addToolApprovalResponse}
        attachments={attachments}
        chatId={id}
        input={input}
        isReadonly={isReadonly}
        messages={messages}
        regenerate={regenerate}
        selectedModelId={currentModelId}
        selectedVisibilityType={visibilityType}
        sendMessage={sendMessage}
        setAttachments={setAttachments}
        setInput={setInput}
        setMessages={setMessages}
        status={status}
        stop={stop}
        votes={votes}
      />

      <AlertDialog
        onOpenChange={setShowCreditCardAlert}
        open={showCreditCardAlert}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Activate AI Gateway</AlertDialogTitle>
            <AlertDialogDescription>
              This application requires{" "}
              {process.env.NODE_ENV === "production" ? "the owner" : "you"} to
              activate Vercel AI Gateway.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                window.open(
                  "https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai%3Fmodal%3Dadd-credit-card",
                  "_blank"
                );
                window.location.href = "/";
              }}
            >
              Activate
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
