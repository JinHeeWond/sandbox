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
  // Ref to store generated image from onData to use in onFinish
  const generatedImageRef = useRef<string | null>(null);

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
      streamProtocol: "ui",
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

      // Handle generated-image data - just store in ref, add in onFinish for reliability
      let generatedImageUrl: string | null = null;

      // Case 1: Direct generated-image data
      if (data.type === "generated-image" && data.imageUrl) {
        console.log("[onData] Case 1: Direct generated-image");
        generatedImageUrl = data.imageUrl;
      }
      // Case 2: Wrapped in data array
      else if (data.type === "data" && Array.isArray(data.data)) {
        const imageData = data.data.find(
          (item: any) => item.type === "generated-image"
        );
        if (imageData?.imageUrl) {
          console.log("[onData] Case 2: generated-image in data array");
          generatedImageUrl = imageData.imageUrl;
        }
      }
      // Case 3: Array containing generated-image object
      else if (Array.isArray(data)) {
        const imageData = data.find(
          (item: any) => item.type === "generated-image"
        );
        if (imageData?.imageUrl) {
          console.log("[onData] Case 3: generated-image in array");
          generatedImageUrl = imageData.imageUrl;
        }
      }

      // URL만 오면 바로 메시지에 추가 (base64가 아닌 URL이므로 빠르게 처리 가능)
      if (generatedImageUrl) {
        console.log("[onData] Found generated image, adding to message immediately");
        generatedImageRef.current = generatedImageUrl;

        // 즉시 메시지에 추가 (clarifying-questions와 동일한 방식)
        setMessages((currentMessages) => {
          const lastMessage = currentMessages.at(-1);
          if (lastMessage?.role === "assistant") {
            const hasImage = lastMessage.parts?.some(
              (p) => (p as any).type === "generated-image"
            );
            if (!hasImage) {
              console.log("[onData] Adding generated-image part to message immediately");
              return currentMessages.map((msg, idx) =>
                idx === currentMessages.length - 1
                  ? {
                      ...msg,
                      parts: [
                        ...(msg.parts || []),
                        {
                          type: "generated-image",
                          imageUrl: generatedImageUrl,
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
      console.log("[onFinish] Stream finished, clarifyingQuestionsRef:", clarifyingQuestionsRef.current, "generatedImageRef:", generatedImageRef.current);

      // Small delay to ensure useChat has finalized its internal state
      await new Promise(resolve => setTimeout(resolve, 100));

      // Clarifying questions 처리
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
      }

      // Generated image 처리 - 스트리밍 완료 후 안전하게 추가
      if (generatedImageRef.current) {
        const imageUrl = generatedImageRef.current;
        generatedImageRef.current = null;

        console.log("[onFinish] Processing generated image in onFinish, imageUrl length:", imageUrl.length);
        setMessages((currentMessages) => {
          const lastMessage = currentMessages.at(-1);
          console.log("[onFinish] Last message for image:", lastMessage?.role, "parts:", lastMessage?.parts?.length);
          if (lastMessage?.role === "assistant") {
            const hasImage = lastMessage.parts?.some(
              (p) => (p as any).type === "generated-image"
            );
            if (!hasImage) {
              console.log("[onFinish] Adding generated-image to message");
              return currentMessages.map((msg, idx) =>
                idx === currentMessages.length - 1
                  ? {
                      ...msg,
                      parts: [
                        ...(msg.parts || []),
                        {
                          type: "generated-image",
                          imageUrl: imageUrl,
                        } as any,
                      ],
                    }
                  : msg
              );
            } else {
              console.log("[onFinish] Message already has generated-image");
            }
          }
          return currentMessages;
        });
      } else {
        // ref에 이미지가 없으면 DB에서 확인
        console.log("[onFinish] No image in ref, checking DB...");
        try {
          const response = await fetch(`/api/chat/${id}/messages`);
          if (response.ok) {
            const dbMessages = await response.json();
            const lastDbMessage = dbMessages.at(-1);
            if (lastDbMessage?.role === "assistant") {
              const imagePart = lastDbMessage.parts?.find(
                (p: any) => p.type === "generated-image"
              );
              const clarifyingPart = lastDbMessage.parts?.find(
                (p: any) => p.type === "clarifying-questions"
              );

              if (imagePart?.imageUrl || clarifyingPart?.questions) {
                console.log("[onFinish] Found special parts in DB");
                setMessages((currentMessages) => {
                  const lastMessage = currentMessages.at(-1);
                  if (lastMessage?.role === "assistant") {
                    const partsToAdd: any[] = [];

                    if (imagePart?.imageUrl) {
                      const hasImage = lastMessage.parts?.some(
                        (p) => (p as any).type === "generated-image"
                      );
                      if (!hasImage) {
                        console.log("[onFinish] Adding image from DB");
                        partsToAdd.push(imagePart);
                      }
                    }

                    if (clarifyingPart?.questions) {
                      const hasQuestions = lastMessage.parts?.some(
                        (p) => (p as any).type === "clarifying-questions"
                      );
                      if (!hasQuestions) {
                        partsToAdd.push(clarifyingPart);
                      }
                    }

                    if (partsToAdd.length > 0) {
                      return currentMessages.map((msg, idx) =>
                        idx === currentMessages.length - 1
                          ? {
                              ...msg,
                              parts: [...(msg.parts || []), ...partsToAdd],
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
    status,
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
