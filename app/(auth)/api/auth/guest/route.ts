import { NextResponse } from "next/server";

// 게스트 로그인 비활성화 - 로그인 페이지로 리다이렉트
export async function GET(request: Request) {
  return NextResponse.redirect(new URL("/login", request.url));
}
