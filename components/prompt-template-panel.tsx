"use client";

import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import type { PromptTemplate } from "@/lib/types";

// 임시 하드코딩된 템플릿 (나중에 API로 대체)
const SAMPLE_TEMPLATES: PromptTemplate[] = [
  {
    id: "1",
    title: "코드 리뷰 요청",
    description: "코드 리뷰를 요청하는 프롬프트",
    content: "[언어]로 작성된 아래 코드를 리뷰해줘:\n\n[코드]",
    category: "개발",
  },
  {
    id: "2",
    title: "버그 분석",
    description: "버그 원인을 분석하는 프롬프트",
    content:
      "다음 에러가 발생했어:\n\n[에러 메시지]\n\n관련 코드:\n[코드]\n\n원인과 해결 방법을 알려줘.",
    category: "개발",
  },
  {
    id: "3",
    title: "개념 설명",
    description: "어려운 개념을 쉽게 설명받는 프롬프트",
    content:
      "[개념]에 대해 [대상]도 이해할 수 있게 쉽게 설명해줘. 예시도 포함해줘.",
    category: "학습",
  },
  {
    id: "4",
    title: "글 요약",
    description: "긴 글을 요약하는 프롬프트",
    content:
      "다음 글을 [문장 수]문장으로 핵심만 요약해줘:\n\n[글 내용]",
    category: "글쓰기",
  },
  {
    id: "5",
    title: "이메일 작성",
    description: "비즈니스 이메일을 작성하는 프롬프트",
    content:
      "[받는 사람]에게 [목적]에 대한 [톤] 톤의 이메일을 작성해줘.\n\n포함할 내용:\n[내용]",
    category: "글쓰기",
  },
];

interface PromptTemplatePanelProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelectTemplate: (template: PromptTemplate) => void;
}

export function PromptTemplatePanel({
  open,
  onOpenChange,
  onSelectTemplate,
}: PromptTemplatePanelProps) {
  // 카테고리별로 그룹화
  const templatesByCategory = SAMPLE_TEMPLATES.reduce(
    (acc, template) => {
      if (!acc[template.category]) {
        acc[template.category] = [];
      }
      acc[template.category].push(template);
      return acc;
    },
    {} as Record<string, PromptTemplate[]>
  );

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-[400px] sm:w-[540px]">
        <SheetHeader>
          <SheetTitle>프롬프트 템플릿</SheetTitle>
          <SheetDescription>
            템플릿을 선택하면 채팅창에 삽입됩니다. [변수] 부분을 수정해서
            사용하세요.
          </SheetDescription>
        </SheetHeader>
        <ScrollArea className="mt-4 h-[calc(100vh-140px)]">
          <div className="flex flex-col gap-6 pr-4">
            {Object.entries(templatesByCategory).map(([category, templates]) => (
              <div key={category}>
                <h3 className="mb-3 text-sm font-medium text-muted-foreground">
                  {category}
                </h3>
                <div className="flex flex-col gap-2">
                  {templates.map((template) => (
                    <button
                      key={template.id}
                      type="button"
                      className="flex flex-col items-start gap-1 rounded-lg border p-3 text-left transition-colors hover:bg-accent"
                      onClick={() => {
                        onSelectTemplate(template);
                        onOpenChange(false);
                      }}
                    >
                      <span className="font-medium">{template.title}</span>
                      <span className="text-sm text-muted-foreground">
                        {template.description}
                      </span>
                      <code className="mt-1 line-clamp-2 text-xs text-muted-foreground">
                        {template.content}
                      </code>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
