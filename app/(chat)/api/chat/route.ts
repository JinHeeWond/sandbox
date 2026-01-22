import { auth, type UserType } from "@/app/(auth)/auth";
import { entitlementsByUserType } from "@/lib/ai/entitlements";
import {
  deleteChatById,
  getChatById,
  getMessageCountByUserId,
  saveChat,
  saveMessages,
} from "@/lib/db/queries";
import { ChatSDKError } from "@/lib/errors";
import { generateUUID } from "@/lib/utils";
import { type PostRequestBody, postRequestBodySchema } from "./schema";
import { createUIMessageStream, createUIMessageStreamResponse } from "ai";

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
    const { id, message, selectedVisibilityType } = requestBody;

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
    } else if (message?.role === "user") {
      // 제목 생성 기능 비활성화 - 사용자 메시지 앞부분을 제목으로 사용
      const userText = message.parts
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
    if (message?.role === "user") {
      await saveMessages({
        messages: [
          {
            chatId: id,
            id: message.id,
            role: "user",
            parts: message.parts,
            attachments: [],
            createdAt: new Date(),
          },
        ],
      });
    }

    // 사용자 메시지 텍스트 추출
    const userMessageText = message?.parts
      ?.filter((part: any) => part.type === "text")
      ?.map((part: any) => part.text)
      ?.join(" ") || "";

    // Python 백엔드 호출
    const pythonResponse = await fetch(`${PYTHON_BACKEND_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: userMessageText,
        thread_id: id,
      }),
    });

    if (!pythonResponse.ok) {
      throw new Error("Python backend error");
    }

    const pythonData = await pythonResponse.json();
    const assistantMessageId = generateUUID();

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
