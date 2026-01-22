"use client";

import * as React from "react";
import { useState } from "react";
import { Download, X, ZoomIn, ZoomOut } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogOverlay,
  DialogPortal,
  DialogClose,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ImagePreviewProps {
  src: string;
  alt?: string;
  className?: string;
  thumbnailClassName?: string;
}

export function ImagePreview({
  src,
  alt = "Image",
  className,
  thumbnailClassName,
}: ImagePreviewProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [scale, setScale] = useState(1);

  const handleZoomIn = () => {
    setScale((prev) => Math.min(prev + 0.25, 3));
  };

  const handleZoomOut = () => {
    setScale((prev) => Math.max(prev - 0.25, 0.5));
  };

  const handleDownload = () => {
    const link = document.createElement("a");
    link.href = src;
    link.download = `image-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const resetZoom = () => {
    setScale(1);
  };

  return (
    <>
      {/* Thumbnail */}
      <div
        className={cn(
          "cursor-pointer overflow-hidden rounded-lg transition-all",
          className
        )}
        onClick={() => setIsOpen(true)}
      >
        <img
          src={src}
          alt={alt}
          className={cn(
            "max-h-[300px] w-auto object-contain transition-transform hover:scale-[1.02]",
            thumbnailClassName
          )}
        />
      </div>

      {/* Preview Modal */}
      <Dialog open={isOpen} onOpenChange={(open) => {
        setIsOpen(open);
        if (!open) resetZoom();
      }}>
        <DialogPortal>
          <DialogOverlay className="bg-black/90" />
          <div className="fixed inset-0 z-50 flex items-center justify-center">
            {/* Toolbar */}
            <div className="fixed top-4 right-4 z-50 flex items-center gap-2">
              <Button
                variant="secondary"
                size="icon"
                onClick={handleZoomOut}
                className="h-9 w-9 rounded-full bg-white/10 hover:bg-white/20"
              >
                <ZoomOut className="h-4 w-4 text-white" />
              </Button>
              <span className="min-w-[60px] text-center text-sm text-white">
                {Math.round(scale * 100)}%
              </span>
              <Button
                variant="secondary"
                size="icon"
                onClick={handleZoomIn}
                className="h-9 w-9 rounded-full bg-white/10 hover:bg-white/20"
              >
                <ZoomIn className="h-4 w-4 text-white" />
              </Button>
              <Button
                variant="secondary"
                size="icon"
                onClick={handleDownload}
                className="h-9 w-9 rounded-full bg-white/10 hover:bg-white/20"
              >
                <Download className="h-4 w-4 text-white" />
              </Button>
              <Button
                variant="secondary"
                size="icon"
                onClick={() => setIsOpen(false)}
                className="h-9 w-9 rounded-full bg-white/10 hover:bg-white/20"
              >
                <X className="h-4 w-4 text-white" />
              </Button>
            </div>

            {/* Image */}
            <div
              className="flex max-h-[90vh] max-w-[90vw] items-center justify-center overflow-auto"
              onClick={() => setIsOpen(false)}
            >
              <img
                src={src}
                alt={alt}
                className="max-h-[85vh] max-w-[85vw] object-contain transition-transform duration-200"
                style={{ transform: `scale(${scale})` }}
                onClick={(e) => e.stopPropagation()}
              />
            </div>
          </div>
        </DialogPortal>
      </Dialog>
    </>
  );
}
