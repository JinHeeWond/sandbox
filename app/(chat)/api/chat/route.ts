import { auth, type UserType } from "@/app/(auth)/auth";
import { entitlementsByUserType } from "@/lib/ai/entitlements";
import {
  deleteChatById,
  getChatById,
  getMessageCountByUserId,
  getMessagesByChatId,
  saveChat,
  saveMessages,
} from "@/lib/db/queries";
import { ChatSDKError } from "@/lib/errors";
import { generateUUID } from "@/lib/utils";
import { type PostRequestBody, postRequestBodySchema } from "./schema";
import { createUIMessageStream, createUIMessageStreamResponse, streamText, generateImage, generateText } from "ai";
import { google } from "@ai-sdk/google";
import { put } from "@vercel/blob";

// 이미지 생성 요청 감지를 위한 system prompt
const INTENT_CLASSIFICATION_PROMPT = `You are an intent classifier. Analyze the user's message and determine if they are requesting IMAGE GENERATION.

IMAGE GENERATION requests include:
- Asking to create, generate, draw, or make an image/picture/illustration
- Describing a scene or object they want visualized
- Requesting artwork, photos, or visual content to be created
- Korean requests like: "그림 그려줘", "이미지 만들어줘", "그려줘", "생성해줘", "만들어줘" (when referring to images)
- Any request that explicitly asks for creating visual content

Respond with ONLY one word:
- "IMAGE_GENERATION" if the user wants an image to be created
- "OTHER" for all other requests (questions, conversations, image analysis, etc.)

Do not explain. Just respond with one word.`;

async function classifyIntent(text: string): Promise<"IMAGE_GENERATION" | "OTHER"> {
  try {
    const { text: classification } = await generateText({
      model: google("gemini-1.5-flash"),
      system: INTENT_CLASSIFICATION_PROMPT,
      prompt: text,
    });

    const result = classification.trim().toUpperCase();
    console.log("[route.ts] Intent classification result:", result);

    if (result.includes("IMAGE_GENERATION")) {
      return "IMAGE_GENERATION";
    }
    return "OTHER";
  } catch (error) {
    console.error("[route.ts] Intent classification error:", error);
    return "OTHER";
  }
}

export const maxDuration = 60;

// Python 백엔드 URL
const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || "http://localhost:8000";

export async function POST(request: Request) {
  let requestBody: PostRequestBody;

  try {
    const json = await request.json();
    requestBody = postRequestBodySchema.parse(json);
  } catch (_) {
    return new ChatSDKError("bad_request:api").toResponse();
  }

  try {
    const { id, message, messages, selectedVisibilityType } = requestBody;

    // 현재 들어온 유저 메시지 결정 (message 우선, 없으면 messages에서 마지막 user)
    const incomingMessage =
      message ??
      [...(messages ?? [])].reverse().find((m: any) => m?.role === "user");

    if (!incomingMessage) {
      return new ChatSDKError("bad_request:api").toResponse();
    }

    // 디버깅: 들어온 메시지 로깅
    console.log("[route.ts] Incoming message:", JSON.stringify(incomingMessage, null, 2));

    const session = await auth();

    if (!session?.user) {
      return new ChatSDKError("unauthorized:chat").toResponse();
    }

    const userType: UserType = session.user.type;

    const messageCount = await getMessageCountByUserId({
      id: session.user.id,
      differenceInHours: 24,
    });

    if (messageCount > entitlementsByUserType[userType].maxMessagesPerDay) {
      return new ChatSDKError("rate_limit:chat").toResponse();
    }

    const chat = await getChatById({ id });

    if (chat) {
      if (chat.userId !== session.user.id) {
        return new ChatSDKError("forbidden:chat").toResponse();
      }
    } else if (incomingMessage?.role === "user") {
      // 제목 생성 기능 비활성화 - 사용자 메시지 앞부분을 제목으로 사용
      const userText = incomingMessage.parts
        ?.filter((part: any) => part.type === "text")
        ?.map((part: any) => part.text)
        ?.join(" ")
        ?.slice(0, 50) || "New Chat";

      await saveChat({
        id,
        userId: session.user.id,
        title: userText,
        visibility: selectedVisibilityType,
      });
    }

    // 사용자 메시지 저장
    if (incomingMessage?.role === "user") {
      await saveMessages({
        messages: [
          {
            chatId: id,
            id: incomingMessage.id,
            role: "user",
            parts: incomingMessage.parts,
            attachments: [],
            createdAt: new Date(),
          },
        ],
      });
    }

    // 사용자 메시지에서 텍스트와 이미지 파트 추출
    const userMessageText = incomingMessage?.parts
      ?.filter((part: any) => part.type === "text")
      ?.map((part: any) => part.text)
      ?.join(" ") || "";

    // 디버깅: 메시지 파트 로깅
    console.log("[route.ts] Message parts:", JSON.stringify(incomingMessage?.parts, null, 2));
    console.log("[route.ts] User message text:", userMessageText);

    const assistantMessageId = generateUUID();

    // ★ 파일 파트 감지를 먼저 수행 (이미지 생성 분류보다 우선)
    const isImagePart = (part: any) => {
      if (part.type !== "file") return false;
      if (part.mediaType?.startsWith("image/")) return true;
      const url = part.url?.toLowerCase() || "";
      return url.endsWith(".jpg") || url.endsWith(".jpeg") || url.endsWith(".png") || url.endsWith(".gif") || url.endsWith(".webp");
    };

    const isPdfPart = (part: any) => {
      if (part.type !== "file") return false;
      if (part.mediaType === "application/pdf") return true;
      const url = part.url?.toLowerCase() || "";
      return url.endsWith(".pdf");
    };

    const imageParts = incomingMessage?.parts?.filter(isImagePart) || [];
    const pdfParts = incomingMessage?.parts?.filter(isPdfPart) || [];
    const hasFilesInCurrentMessage = imageParts.length > 0 || pdfParts.length > 0;

    console.log("[route.ts] Image parts found:", imageParts.length);
    console.log("[route.ts] PDF parts found:", pdfParts.length);

    // ★ 파일이 첨부되어 있으면 이미지 생성 분류를 건너뛰고 멀티모달 처리로 이동
    // 이미지가 첨부된 상태에서 "이게 무슨 이미지냐"는 이미지 분석 요청이지 이미지 생성 요청이 아님
    if (hasFilesInCurrentMessage) {
      console.log("[route.ts] Files attached, skipping image generation check - proceeding to multimodal processing");
    }

    // ★ 키워드 기반 이미지 생성 요청 1차 체크 (LLM 분류보다 안정적)
    // 단, 파일이 첨부되어 있으면 이미지 생성으로 분류하지 않음
    const looksLikeImageGen = (text: string) => {
      const t = (text || "").trim();
      if (!t) return false;
      // "이미지"만 있는 경우는 제외하고, 명확한 생성 요청만 매칭
      // "이미지 생성", "이미지 만들어", "그림 그려줘" 등
      return /(이미지\s*(를\s*)?(생성|만들|그려)|그림\s*(을\s*)?(그려|만들|생성)|일러스트\s*(를\s*)?(그려|만들|생성)|사진\s*(을\s*)?(만들|생성)|포스터\s*(를\s*)?(만들|생성)|짤\s*(을\s*)?(만들|생성)|그려\s*줘|그려\s*주세요|만들어\s*줘|만들어\s*주세요|생성해\s*줘|생성해\s*주세요|draw|generate\s+(an?\s+)?image|create\s+(an?\s+)?image|make\s+(an?\s+)?image|make\s+(a\s+)?picture)/i.test(t);
    };

    // ★ 이미지 생성 요청 체크 - 파일이 첨부되어 있으면 건너뜀
    let intent: "IMAGE_GENERATION" | "OTHER" = "OTHER";
    if (!hasFilesInCurrentMessage) {
      if (looksLikeImageGen(userMessageText)) {
        intent = "IMAGE_GENERATION";
        console.log("[route.ts] Intent matched by keyword regex: IMAGE_GENERATION");
      } else {
        intent = await classifyIntent(userMessageText);
      }
      console.log("[route.ts] Intent classification:", intent, "for text:", userMessageText);
    }

    if (intent === "IMAGE_GENERATION" && !hasFilesInCurrentMessage) {
      console.log("[route.ts] Image generation request detected, using Imagen");

      // ★ 핵심 수정: 이미지 생성을 execute 밖에서 먼저 수행
      // 이렇게 하면 execute 함수가 빨리 완료되어 HTTP 연결이 빨리 닫힘
      let imageUrl: string | null = null;
      let responseText = "이미지를 생성했습니다.";
      let isError = false;

      try {
        // 이미지 생성 (google.image() 사용) - Imagen 4 모델 사용
        const { image } = await generateImage({
          model: google.image("imagen-4.0-fast-generate-001"),
          prompt: userMessageText,
        });

        // 1) base64 -> Buffer
        const pngBuffer = Buffer.from(image.base64, "base64");

        // 2) Vercel Blob에 업로드 (파일명은 유니크하게)
        const blob = await put(
          `generated/${id}/${assistantMessageId}.png`,
          pngBuffer,
          { access: "public", contentType: "image/png" }
        );

        imageUrl = blob.url;
        console.log("[route.ts] Image uploaded to Vercel Blob:", imageUrl);

        // DB 저장은 비동기로 (절대 await 하지 않음)
        void saveMessages({
          messages: [
            {
              chatId: id,
              id: assistantMessageId,
              role: "assistant",
              parts: [
                { type: "text", text: responseText },
                { type: "generated-image", imageUrl },
              ] as any,
              attachments: [],
              createdAt: new Date(),
            },
          ],
        }).catch(console.error);

      } catch (error) {
        console.error("[route.ts] Imagen error:", error);
        responseText = "이미지 생성 중 오류가 발생했습니다. 다시 시도해주세요.";
        isError = true;
      }

      // ★ execute 함수는 이미 준비된 데이터만 빠르게 전송하고 종료
      const stream = createUIMessageStream({
        execute: async ({ writer }) => {
          writer.write({ type: "start", messageId: assistantMessageId } as any);
          writer.write({ type: "start-step" } as any);
          writer.write({ type: "text-start", id: assistantMessageId });
          writer.write({ type: "text-delta", delta: responseText, id: assistantMessageId });

          if (imageUrl) {
            writer.write({
              type: "data",
              data: [{ type: "generated-image", imageUrl }],
            } as any);
          }

          writer.write({ type: "text-end", id: assistantMessageId });
          writer.write({ type: "finish-step" } as any);
          writer.write({ type: "finish", finishReason: isError ? "error" : "stop" } as any);
          // execute 함수가 여기서 끝나면 스트림 자동 종료 → HTTP 연결 닫힘 → status "ready"
        },
        onError: (error: unknown) => {
          console.error("[route.ts] Image generation stream error:", error);
          return "이미지 생성 중 오류가 발생했습니다.";
        },
      });

      return createUIMessageStreamResponse({ stream });
    }

    // 이전 메시지들을 조회하여 파일 컨텍스트 확인
    const previousMessages = await getMessagesByChatId({ id });

    // ★ hasFiles는 현재 메시지에 파일이 있을 때만 true (히스토리는 멀티모달 분기에 영향 안 줌)
    const hasFiles = hasFilesInCurrentMessage;

    if (imageParts.length > 0) {
      console.log("[route.ts] Image parts detail:", JSON.stringify(imageParts, null, 2));
    }
    console.log("[route.ts] Files in current message:", hasFilesInCurrentMessage);

    // 파일 파트를 처리하는 헬퍼 함수
    const processFileParts = async (parts: any[]) => {
      const contentParts: any[] = [];

      for (const part of parts) {
        if (part.type === "text" && part.text) {
          contentParts.push({ type: "text", text: part.text });
        } else if (isImagePart(part)) {
          const imageUrl = part.url;
          try {
            if (imageUrl.startsWith("data:")) {
              contentParts.push({ type: "image", image: imageUrl });
            } else if (!imageUrl.startsWith("blob:")) {
              const response = await fetch(imageUrl);
              if (response.ok) {
                const arrayBuffer = await response.arrayBuffer();
                const contentType = response.headers.get("content-type") || "image/png";
                contentParts.push({
                  type: "image",
                  image: Buffer.from(arrayBuffer),
                  mimeType: contentType,
                });
              }
            }
          } catch (error) {
            console.error("[route.ts] Failed to process image:", error);
          }
        } else if (isPdfPart(part)) {
          const pdfUrl = part.url;
          try {
            if (pdfUrl.startsWith("data:")) {
              const match = pdfUrl.match(/^data:application\/pdf;base64,(.+)$/);
              if (match) {
                const pdfBuffer = Buffer.from(match[1], "base64");
                contentParts.push({
                  type: "file",
                  data: pdfBuffer,
                  mediaType: "application/pdf",
                });
              }
            } else if (!pdfUrl.startsWith("blob:")) {
              const response = await fetch(pdfUrl);
              if (response.ok) {
                const arrayBuffer = await response.arrayBuffer();
                contentParts.push({
                  type: "file",
                  data: Buffer.from(arrayBuffer),
                  mediaType: "application/pdf",
                });
              }
            }
          } catch (error) {
            console.error("[route.ts] Failed to process PDF:", error);
          }
        }
      }

      return contentParts;
    };

    // 현재 메시지에 이미지나 PDF가 있으면 Gemini로 직접 처리 (멀티모달)
    if (hasFiles) {
      console.log("[route.ts] Multimodal request detected, using Gemini");
      console.log("[route.ts] Current message - Images:", imageParts.length, "PDFs:", pdfParts.length);

      try {
        // 전체 대화 히스토리를 Gemini messages 형식으로 변환
        const geminiMessages: any[] = [];

        // 이전 메시지들 처리
        for (const prevMsg of previousMessages) {
          const parts = prevMsg.parts as any[];
          if (!parts || parts.length === 0) continue;

          const role = prevMsg.role === "user" ? "user" : "assistant";

          // 이전 메시지의 파일과 텍스트를 처리
          const contentParts = await processFileParts(parts);

          if (contentParts.length > 0) {
            geminiMessages.push({
              role,
              content: contentParts,
            });
          }
        }

        // 현재 메시지 처리
        const currentContentParts: any[] = [];

        // 텍스트 추가
        if (userMessageText) {
          currentContentParts.push({ type: "text", text: userMessageText });
        } else if (pdfParts.length > 0) {
          currentContentParts.push({ type: "text", text: "이 PDF 문서를 분석해주세요." });
        } else if (imageParts.length > 0) {
          currentContentParts.push({ type: "text", text: "이 이미지를 분석해주세요." });
        } else {
          // 파일이 없는 후속 질문인 경우
          currentContentParts.push({ type: "text", text: userMessageText || "계속해주세요." });
        }

        // 현재 메시지의 파일들 처리
        const currentFileParts = await processFileParts([...imageParts, ...pdfParts]);
        currentContentParts.push(...currentFileParts);

        // 현재 메시지를 geminiMessages에 추가
        geminiMessages.push({
          role: "user",
          content: currentContentParts,
        });

        console.log("[route.ts] Sending to Gemini with", geminiMessages.length, "messages");
        console.log("[route.ts] Message roles:", geminiMessages.map(m => m.role));

        // streamText로 Gemini 호출 - gemini-2.0-flash는 PDF 분석도 지원
        const modelId = "gemini-2.0-flash";
        console.log("[route.ts] Using model:", modelId);

        const result = streamText({
          model: google(modelId),
          messages: geminiMessages,
          onFinish: async ({ text }) => {
            console.log("[route.ts] Gemini finished, text length:", text.length);
            // DB에 저장
            try {
              await saveMessages({
                messages: [
                  {
                    chatId: id,
                    id: assistantMessageId,
                    role: "assistant",
                    parts: [{ type: "text", text }] as any,
                    attachments: [],
                    createdAt: new Date(),
                  },
                ],
              });
              console.log("[route.ts] Message saved to DB");
            } catch (error) {
              console.error("[route.ts] Failed to save message:", error);
            }
          },
        });

        // toUIMessageStream()을 사용하여 UI message stream 형식으로 변환
        return createUIMessageStreamResponse({
          stream: result.toUIMessageStream(),
        });
      } catch (error) {
        console.error("[route.ts] Error in multimodal processing:", error);
        // 에러 발생 시 에러 메시지 스트리밍
        const errorType = pdfParts.length > 0 ? "PDF" : "이미지";
        const stream = createUIMessageStream({
          execute: async ({ writer }) => {
            // 스트림 시작 이벤트
            writer.write({ type: "start", messageId: assistantMessageId } as any);
            writer.write({ type: "start-step" } as any);

            writer.write({ type: "text-start", id: assistantMessageId });
            const errorText = `${errorType} 분석 중 오류가 발생했습니다. 다시 시도해주세요. (${error instanceof Error ? error.message : "Unknown error"})`;
            writer.write({ type: "text-delta", delta: errorText, id: assistantMessageId });
            writer.write({ type: "text-end", id: assistantMessageId });

            // 스트림 종료 이벤트 - finishReason 필수!
            writer.write({ type: "finish-step" } as any);
            writer.write({ type: "finish", finishReason: "error" } as any);
          },
          onError: (err: unknown) => {
            console.error("[route.ts] Stream error:", err);
            return "오류가 발생했습니다.";
          },
        });
        return createUIMessageStreamResponse({ stream });
      }
    }

    // 텍스트만 있으면 Python 백엔드 호출
    console.log("[route.ts] Text-only request, using Python backend");

    // 이전 대화 히스토리를 Python 백엔드에 전달
    const conversationHistory = previousMessages.map(msg => ({
      role: msg.role,
      content: (msg.parts as any[])
        ?.filter((part: any) => part.type === "text")
        ?.map((part: any) => part.text)
        ?.join(" ") || "",
    })).filter(msg => msg.content.trim() !== "");

    const pythonResponse = await fetch(`${PYTHON_BACKEND_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userMessageText,
        thread_id: id,
        history: conversationHistory,
      }),
    });

    if (!pythonResponse.ok) {
      throw new Error("Python backend error");
    }

    const pythonData = await pythonResponse.json();

    // clarifying_questions일 때는 텍스트를 비워서 클릭 가능한 UI만 표시
    const isClarifyingQuestions = pythonData.response_type === "clarifying_questions";
    const responseText = isClarifyingQuestions ? "" : (pythonData.response || "");

    // Build message parts
    const messageParts: Array<{ type: string; text?: string; questions?: any[] }> = [];

    // 텍스트가 있을 때만 텍스트 파트 추가
    if (responseText) {
      messageParts.push({ type: "text", text: responseText });
    }

    // Add clarifying questions data if present
    if (pythonData.response_type === "clarifying_questions" && pythonData.questions) {
      messageParts.push({
        type: "clarifying-questions",
        questions: pythonData.questions.map((q: any, idx: number) => ({
          id: `q-${idx}`,
          question: q.question,
          choices: (q.choices || []).map((c: string, cIdx: number) => ({
            id: `q-${idx}-c-${cIdx}`,
            text: c,
          })),
          allowMultiple: q.allowMultiple ?? false,
        })),
      });
    }

    // DB에 저장 (비동기로 처리)
    saveMessages({
      messages: [
        {
          chatId: id,
          id: assistantMessageId,
          role: "assistant",
          parts: messageParts as any,
          attachments: [],
          createdAt: new Date(),
        },
      ],
    }).catch(console.error);

    // AI SDK v6의 createUIMessageStream을 사용하여 스트리밍 응답 생성
    const stream = createUIMessageStream({
      execute: async ({ writer }) => {
        // 스트림 시작 이벤트 (useChat이 응답 시작을 인식)
        writer.write({ type: "start", messageId: assistantMessageId } as any);
        writer.write({ type: "start-step" } as any);

        // text-start로 새 텍스트 파트 시작
        writer.write({ type: "text-start", id: assistantMessageId });

        // 텍스트가 있을 때만 text-delta 전송
        if (responseText) {
          writer.write({ type: "text-delta", delta: responseText, id: assistantMessageId });
        }

        writer.write({ type: "text-end", id: assistantMessageId });

        // Send clarifying questions part if present - 여러 형식으로 전송하여 호환성 확보
        if (isClarifyingQuestions && pythonData.questions) {
          const questionsData = pythonData.questions.map((q: any, idx: number) => ({
            id: `q-${idx}`,
            question: q.question,
            choices: (q.choices || []).map((c: string, cIdx: number) => ({
              id: `q-${idx}-c-${cIdx}`,
              text: c,
            })),
            allowMultiple: q.allowMultiple ?? false,
          }));

          console.log("[route.ts] Sending clarifying questions:", JSON.stringify(questionsData));

          // 방법 1: data 배열로 전송
          writer.write({
            type: "data",
            data: [{
              type: "clarifying-questions",
              questions: questionsData,
            }],
          } as any);

          // 방법 2: 직접 객체로도 전송 (클라이언트에서 여러 형식 처리)
          writer.write({
            type: "clarifying-questions",
            questions: questionsData,
          } as any);
        }

        // 스트림 종료 이벤트 - finishReason 필수!
        writer.write({ type: "finish-step" } as any);
        writer.write({ type: "finish", finishReason: "stop" } as any);
      },
      onError: (error: unknown) => {
        console.error("Stream error:", error);
        return "An error occurred";
      },
    });

    return createUIMessageStreamResponse({ stream });

  } catch (error) {
    console.error("Error in chat API:", error);
    return new ChatSDKError("offline:chat").toResponse();
  }
}

export async function DELETE(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get("id");

  if (!id) {
    return new ChatSDKError("bad_request:api").toResponse();
  }

  const session = await auth();

  if (!session?.user) {
    return new ChatSDKError("unauthorized:chat").toResponse();
  }

  const chat = await getChatById({ id });

  if (chat?.userId !== session.user.id) {
    return new ChatSDKError("forbidden:chat").toResponse();
  }

  const deletedChat = await deleteChatById({ id });

  return Response.json(deletedChat, { status: 200 });
}
