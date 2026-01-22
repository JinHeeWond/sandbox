"use client";

import * as React from "react";
import { useState } from "react";
import { Download, X, ZoomIn, ZoomOut, ChevronLeft, ChevronRight } from "lucide-react";
import {
  Dialog,
  DialogOverlay,
  DialogPortal,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ImageItem {
  src: string;
  alt?: string;
}

interface ImageGalleryProps {
  images: ImageItem[];
  className?: string;
}

export function ImageGallery({ images, className }: ImageGalleryProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [scale, setScale] = useState(1);

  const handleOpen = (index: number) => {
    setCurrentIndex(index);
    setScale(1);
    setIsOpen(true);
  };

  const handleClose = () => {
    setIsOpen(false);
    setScale(1);
  };

  const handlePrev = () => {
    setCurrentIndex((prev) => (prev > 0 ? prev - 1 : images.length - 1));
    setScale(1);
  };

  const handleNext = () => {
    setCurrentIndex((prev) => (prev < images.length - 1 ? prev + 1 : 0));
    setScale(1);
  };

  const handleZoomIn = () => {
    setScale((prev) => Math.min(prev + 0.25, 3));
  };

  const handleZoomOut = () => {
    setScale((prev) => Math.max(prev - 0.25, 0.5));
  };

  const handleDownload = () => {
    const currentImage = images[currentIndex];
    const link = document.createElement("a");
    link.href = currentImage.src;
    link.download = `image-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowLeft") handlePrev();
    if (e.key === "ArrowRight") handleNext();
    if (e.key === "Escape") handleClose();
  };

  if (images.length === 0) return null;

  // Single image
  if (images.length === 1) {
    return (
      <>
        <div
          className={cn(
            "cursor-pointer overflow-hidden rounded-lg border shadow-sm transition-all hover:shadow-md",
            className
          )}
          onClick={() => handleOpen(0)}
        >
          <img
            src={images[0].src}
            alt={images[0].alt || "Image"}
            className="max-h-[300px] w-auto object-contain transition-transform hover:scale-[1.02]"
          />
        </div>
        {renderModal()}
      </>
    );
  }

  // Multiple images - Grid Gallery
  return (
    <>
      <div className={cn("grid gap-2", className, {
        "grid-cols-2": images.length === 2,
        "grid-cols-2 md:grid-cols-3": images.length >= 3,
      })}>
        {images.map((image, index) => (
          <div
            key={index}
            className="relative cursor-pointer overflow-hidden rounded-lg border shadow-sm transition-all hover:shadow-md"
            onClick={() => handleOpen(index)}
          >
            <img
              src={image.src}
              alt={image.alt || `Image ${index + 1}`}
              className="h-[150px] w-full object-cover transition-transform hover:scale-[1.05]"
            />
            {images.length > 4 && index === 3 && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                <span className="text-2xl font-bold text-white">
                  +{images.length - 4}
                </span>
              </div>
            )}
          </div>
        )).slice(0, 4)}
      </div>
      {renderModal()}
    </>
  );

  function renderModal() {
    const currentImage = images[currentIndex];

    return (
      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogPortal>
          <DialogOverlay className="bg-black/90" />
          <div
            className="fixed inset-0 z-50 flex items-center justify-center outline-none"
            onKeyDown={handleKeyDown}
            tabIndex={0}
          >
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
                onClick={handleClose}
                className="h-9 w-9 rounded-full bg-white/10 hover:bg-white/20"
              >
                <X className="h-4 w-4 text-white" />
              </Button>
            </div>

            {/* Image counter */}
            <div className="fixed top-4 left-4 z-50">
              <span className="text-sm text-white">
                {currentIndex + 1} / {images.length}
              </span>
            </div>

            {/* Navigation - Previous */}
            {images.length > 1 && (
              <Button
                variant="secondary"
                size="icon"
                onClick={handlePrev}
                className="fixed left-4 top-1/2 z-50 h-12 w-12 -translate-y-1/2 rounded-full bg-white/10 hover:bg-white/20"
              >
                <ChevronLeft className="h-6 w-6 text-white" />
              </Button>
            )}

            {/* Image */}
            <div
              className="flex max-h-[90vh] max-w-[90vw] items-center justify-center overflow-auto"
              onClick={handleClose}
            >
              <img
                src={currentImage.src}
                alt={currentImage.alt || "Image"}
                className="max-h-[85vh] max-w-[85vw] object-contain transition-transform duration-200"
                style={{ transform: `scale(${scale})` }}
                onClick={(e) => e.stopPropagation()}
              />
            </div>

            {/* Navigation - Next */}
            {images.length > 1 && (
              <Button
                variant="secondary"
                size="icon"
                onClick={handleNext}
                className="fixed right-4 top-1/2 z-50 h-12 w-12 -translate-y-1/2 rounded-full bg-white/10 hover:bg-white/20"
              >
                <ChevronRight className="h-6 w-6 text-white" />
              </Button>
            )}

            {/* Thumbnails */}
            {images.length > 1 && (
              <div className="fixed bottom-4 left-1/2 z-50 flex -translate-x-1/2 gap-2 rounded-lg bg-black/50 p-2">
                {images.map((image, index) => (
                  <div
                    key={index}
                    className={cn(
                      "h-12 w-12 cursor-pointer overflow-hidden rounded border-2 transition-all",
                      currentIndex === index
                        ? "border-white"
                        : "border-transparent opacity-60 hover:opacity-100"
                    )}
                    onClick={() => {
                      setCurrentIndex(index);
                      setScale(1);
                    }}
                  >
                    <img
                      src={image.src}
                      alt={image.alt || `Thumbnail ${index + 1}`}
                      className="h-full w-full object-cover"
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        </DialogPortal>
      </Dialog>
    );
  }
}
