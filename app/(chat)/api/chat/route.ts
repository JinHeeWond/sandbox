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

    const responseText = pythonData.response || "";

    // DB에 저장 (비동기로 처리)
    saveMessages({
      messages: [
        {
          chatId: id,
          id: assistantMessageId,
          role: "assistant",
          parts: [{ type: "text", text: responseText }],
          attachments: [],
          createdAt: new Date(),
        },
      ],
    }).catch(console.error);

    // AI SDK v6의 createUIMessageStream을 사용하여 스트리밍 응답 생성
    const stream = createUIMessageStream({
      execute: async ({ writer }) => {
        writer.write({ type: "text-start", id: assistantMessageId });
        writer.write({ type: "text-delta", delta: responseText, id: assistantMessageId });
        writer.write({ type: "text-end", id: assistantMessageId });
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
