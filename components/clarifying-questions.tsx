"use client";

import { useState } from "react";
import type { UseChatHelpers } from "@ai-sdk/react";
import { ChevronRight, PenLine, Send, Check, CheckCircle2, Circle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { ChatMessage, ClarifyingQuestion } from "@/lib/types";
import { cn } from "@/lib/utils";

interface ClarifyingQuestionsProps {
  questions: ClarifyingQuestion[];
  chatId: string;
  sendMessage: UseChatHelpers<ChatMessage>["sendMessage"];
  isDisabled?: boolean;
}

export function ClarifyingQuestions({
  questions,
  chatId: _chatId,
  sendMessage,
  isDisabled = false,
}: ClarifyingQuestionsProps) {
  // Current question index
  const [currentIndex, setCurrentIndex] = useState(0);
  // Track selected choices for each question: { [questionId]: selectedAnswers[] }
  const [answers, setAnswers] = useState<Record<string, string[]>>({});
  // Track custom input mode for current question
  const [isCustomInput, setIsCustomInput] = useState(false);
  // Track custom input value for current question
  const [customInputValue, setCustomInputValue] = useState("");
  // Track if answers have been submitted
  const [isSubmitted, setIsSubmitted] = useState(false);

  const currentQuestion = questions[currentIndex];
  const isLastQuestion = currentIndex === questions.length - 1;
  const totalQuestions = questions.length;
  const allowMultiple = currentQuestion?.allowMultiple ?? false;
  const currentSelections = answers[currentQuestion?.id] || [];

  const handleChoiceSelect = (choiceText: string) => {
    if (isSubmitted || isDisabled) return;

    if (allowMultiple) {
      // Multi-select: toggle the choice
      setAnswers((prev) => {
        const current = prev[currentQuestion.id] || [];
        if (current.includes(choiceText)) {
          return { ...prev, [currentQuestion.id]: current.filter((t) => t !== choiceText) };
        } else {
          return { ...prev, [currentQuestion.id]: [...current, choiceText] };
        }
      });
      // Clear custom input mode if selecting a choice
      setIsCustomInput(false);
      setCustomInputValue("");
    } else {
      // Single-select: save and move to next
      const newAnswers = {
        ...answers,
        [currentQuestion.id]: [choiceText],
      };
      setAnswers(newAnswers);

      // Reset custom input state
      setIsCustomInput(false);
      setCustomInputValue("");

      // Move to next question or submit
      if (isLastQuestion) {
        submitAllAnswers(newAnswers);
      } else {
        setCurrentIndex(currentIndex + 1);
      }
    }
  };

  const handleMultiSelectNext = () => {
    if (isSubmitted || isDisabled || currentSelections.length === 0) return;

    if (isLastQuestion) {
      submitAllAnswers(answers);
    } else {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const handleCustomInputToggle = () => {
    if (isSubmitted || isDisabled) return;
    setIsCustomInput(true);
    // Clear selections when switching to custom input
    if (!allowMultiple) {
      setAnswers((prev) => ({ ...prev, [currentQuestion.id]: [] }));
    }
  };

  const handleCustomInputSubmit = () => {
    if (isSubmitted || isDisabled || !customInputValue.trim()) return;

    let newAnswers: Record<string, string[]>;

    if (allowMultiple) {
      // For multi-select, add custom input to existing selections
      const current = answers[currentQuestion.id] || [];
      newAnswers = {
        ...answers,
        [currentQuestion.id]: [...current, customInputValue.trim()],
      };
    } else {
      // For single-select, replace with custom input
      newAnswers = {
        ...answers,
        [currentQuestion.id]: [customInputValue.trim()],
      };
    }

    setAnswers(newAnswers);

    // Reset custom input state
    setIsCustomInput(false);
    setCustomInputValue("");

    // For multi-select, don't auto-advance (user might want to add more)
    if (allowMultiple) {
      return;
    }

    // Move to next question or submit (single-select only)
    if (isLastQuestion) {
      submitAllAnswers(newAnswers);
    } else {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const submitAllAnswers = (finalAnswers: Record<string, string[]>) => {
    const answerParts: string[] = [];

    questions.forEach((q) => {
      const answerList = finalAnswers[q.id];
      if (answerList && answerList.length > 0) {
        answerParts.push(`${q.question}: ${answerList.join(", ")}`);
      }
    });

    const answerText = answerParts.join("\n");
    if (answerText) {
      setIsSubmitted(true);
      sendMessage({
        role: "user",
        parts: [{ type: "text", text: answerText }],
      });
    }
  };

  const formatAnswerDisplay = (answerList: string[] | undefined) => {
    if (!answerList || answerList.length === 0) return "";
    return answerList.join(", ");
  };

  // Show completed state
  if (isSubmitted) {
    return (
      <div className="flex flex-col gap-3 w-full max-w-md">
        <div className="rounded-lg border border-green-200 bg-green-50 dark:bg-green-950/30 dark:border-green-800 p-4">
          <div className="flex items-center gap-2 text-green-700 dark:text-green-400">
            <Check className="size-5" />
            <span className="font-medium">답변이 전송되었습니다</span>
          </div>
          <div className="mt-3 space-y-2 text-sm text-muted-foreground">
            {questions.map((q) => (
              <div key={q.id}>
                <span className="font-medium">{q.question}</span>
                <span className="ml-2">{formatAnswerDisplay(answers[q.id])}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4 w-full max-w-md">
      {/* Progress indicator */}
      <div className="flex items-center gap-2">
        <div className="flex gap-1">
          {questions.map((_, idx) => (
            <div
              key={idx}
              className={cn(
                "h-1.5 w-6 rounded-full transition-colors",
                idx < currentIndex
                  ? "bg-primary"
                  : idx === currentIndex
                  ? "bg-primary/60"
                  : "bg-muted"
              )}
            />
          ))}
        </div>
        <span className="text-xs text-muted-foreground">
          {currentIndex + 1} / {totalQuestions}
        </span>
      </div>

      {/* Current question */}
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-medium text-base text-foreground">
            {currentQuestion.question}
          </h4>
          {allowMultiple && (
            <span className="text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded">
              복수 선택
            </span>
          )}
        </div>

        {!isCustomInput ? (
          <div className="flex flex-col gap-2">
            {currentQuestion.choices.map((choice, idx) => {
              const isSelected = currentSelections.includes(choice.text);

              return (
                <button
                  key={choice.id}
                  type="button"
                  onClick={() => handleChoiceSelect(choice.text)}
                  disabled={isDisabled}
                  className={cn(
                    "flex items-center justify-between w-full rounded-lg border px-4 py-3 text-sm text-left transition-all",
                    isSelected
                      ? "border-primary bg-primary/10 text-foreground"
                      : "border-border bg-background text-foreground hover:border-primary hover:bg-primary/5",
                    isDisabled && "cursor-not-allowed opacity-60"
                  )}
                >
                  <span className="flex items-center gap-3">
                    {allowMultiple ? (
                      isSelected ? (
                        <CheckCircle2 className="size-5 text-primary" />
                      ) : (
                        <Circle className="size-5 text-muted-foreground" />
                      )
                    ) : (
                      <span className={cn(
                        "flex items-center justify-center size-6 rounded-full text-xs font-medium",
                        isSelected ? "bg-primary text-primary-foreground" : "bg-muted"
                      )}>
                        {idx + 1}
                      </span>
                    )}
                    {choice.text}
                  </span>
                  {!allowMultiple && (
                    <ChevronRight className="size-4 text-muted-foreground" />
                  )}
                </button>
              );
            })}

            {/* Custom input option */}
            <button
              type="button"
              onClick={handleCustomInputToggle}
              disabled={isDisabled}
              className={cn(
                "flex items-center justify-between w-full rounded-lg border border-dashed px-4 py-3 text-sm text-left transition-all",
                "border-border bg-background text-muted-foreground",
                "hover:border-primary hover:text-foreground hover:bg-primary/5",
                isDisabled && "cursor-not-allowed opacity-60"
              )}
            >
              <span className="flex items-center gap-3">
                <span className="flex items-center justify-center size-6 rounded-full bg-muted">
                  <PenLine className="size-3" />
                </span>
                직접 입력하기
              </span>
              <ChevronRight className="size-4" />
            </button>

            {/* Next button for multi-select */}
            {allowMultiple && (
              <div className="flex justify-end mt-2">
                <Button
                  size="sm"
                  onClick={handleMultiSelectNext}
                  disabled={isDisabled || currentSelections.length === 0}
                  className="gap-1"
                >
                  {isLastQuestion ? (
                    <>
                      <Send className="size-3" />
                      전송
                    </>
                  ) : (
                    <>
                      다음
                      <ChevronRight className="size-3" />
                    </>
                  )}
                </Button>
              </div>
            )}
          </div>
        ) : (
          <div className="flex flex-col gap-3">
            <Input
              placeholder="직접 입력해주세요..."
              value={customInputValue}
              onChange={(e) => setCustomInputValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && customInputValue.trim()) {
                  handleCustomInputSubmit();
                }
              }}
              disabled={isDisabled}
              autoFocus
              className="text-sm"
            />
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setIsCustomInput(false);
                  setCustomInputValue("");
                }}
                disabled={isDisabled}
              >
                취소
              </Button>
              <Button
                size="sm"
                onClick={handleCustomInputSubmit}
                disabled={isDisabled || !customInputValue.trim()}
                className="gap-1"
              >
                {allowMultiple ? (
                  "추가"
                ) : isLastQuestion ? (
                  <>
                    <Send className="size-3" />
                    전송
                  </>
                ) : (
                  <>
                    다음
                    <ChevronRight className="size-3" />
                  </>
                )}
              </Button>
            </div>

            {/* Show current selections for multi-select */}
            {allowMultiple && currentSelections.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-1">
                {currentSelections.map((sel, idx) => (
                  <span
                    key={idx}
                    className="inline-flex items-center gap-1 text-xs bg-primary/10 text-primary px-2 py-1 rounded-full"
                  >
                    <Check className="size-3" />
                    {sel}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Show current selections for multi-select (when not in custom input mode) */}
        {!isCustomInput && allowMultiple && currentSelections.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-3 pt-3 border-t border-border">
            <span className="text-xs text-muted-foreground mr-1">선택됨:</span>
            {currentSelections.map((sel, idx) => (
              <span
                key={idx}
                className="inline-flex items-center gap-1 text-xs bg-primary/10 text-primary px-2 py-1 rounded-full"
              >
                {sel}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Previously answered questions summary */}
      {currentIndex > 0 && (
        <div className="text-xs text-muted-foreground space-y-1">
          {questions.slice(0, currentIndex).map((q) => (
            <div key={q.id} className="flex gap-2">
              <Check className="size-3 text-primary shrink-0 mt-0.5" />
              <span>
                <span className="font-medium">{q.question}</span>{" "}
                <span className="text-foreground">{formatAnswerDisplay(answers[q.id])}</span>
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
