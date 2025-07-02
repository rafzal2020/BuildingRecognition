"use client"

import type React from "react"

import { useState } from "react"
import { Upload, Building2, Info, Calendar, Ruler, Layers } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface BuildingData {
  building_name: string
  confidence?: number
  wikipedia_info?: {
    description: string
    wikipedia_link: string
    image_url: string
  }
  height?: string
  floors?: string
  status?: string
  completed?: string
  location?: string
}

export default function BuildingRecognition() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [data, setData] = useState<BuildingData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [dragActive, setDragActive] = useState(false)

  const handleFileChange = (file: File) => {
    setSelectedImage(file)
    setError(null)
    setData(null)

    // Create preview
    const reader = new FileReader()
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string)
    }
    reader.readAsDataURL(file)
  }

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) handleFileChange(file)
  }

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith("image/")) {
      handleFileChange(file)
    }
  }

  const handleUpload = async () => {
    if (!selectedImage) {
      setError("Please select an image first.")
      return
    }

    const formData = new FormData()
    formData.append("image", selectedImage)

    setLoading(true)
    setError(null)

    try {
      // Connect to your Python backend
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error("Failed to identify building")
      }

      const result = await response.json()
      console.log(result)
      if (result.building_name === "Unknown") {
        setError("Could not identify the building. Please try another image.")
        setData(null) // Optionally clear previous data
      } else {
        setData(result)
        setError(null)
      }
    } catch (err) {
      setError("Failed to identify building. Please try again.")
      console.error("Upload error:", err)
    } finally {
      setLoading(false)
    }
  }

  const resetApp = () => {
    setSelectedImage(null)
    setImagePreview(null)
    setData(null)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-dark via-primary-dark/95 to-secondary-dark">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Building2 className="h-10 w-10 text-accent-primary" />
            <h1 className="text-4xl font-bold text-neutral-light">BuildingRecognition</h1>
          </div>
          <p className="text-xl text-neutral-medium max-w-2xl mx-auto">
            Upload an image of any building and discover its history, architecture, and fascinating details
          </p>
        </div>

        <div className="max-w-6xl mx-auto">
          {!data ? (
            /* Upload Section */
            <Card className="bg-neutral-dark/50 border-neutral-medium/20 backdrop-blur-sm">
              <CardHeader className="text-center">
                <CardTitle className="text-2xl text-neutral-light">Identify Your Building</CardTitle>
                <CardDescription className="text-neutral-medium">
                  Drag and drop an image or click to browse
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Drag & Drop Area */}
                <div
                  className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 ${
                    dragActive
                      ? "border-accent-primary bg-accent-primary/10"
                      : "border-neutral-medium/30 hover:border-accent-primary/50"
                  }`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleInputChange}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                  />

                  {imagePreview ? (
                    <div className="space-y-4">
                      <img
                        src={imagePreview || "/placeholder.svg"}
                        alt="Preview"
                        className="max-w-full max-h-64 mx-auto rounded-lg shadow-lg"
                      />
                      <p className="text-neutral-medium">{selectedImage?.name}</p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <Upload className="h-12 w-12 text-neutral-medium mx-auto" />
                      <div>
                        <p className="text-lg text-neutral-light mb-2">Drop your building image here</p>
                        <p className="text-neutral-medium">or click to browse files</p>
                      </div>
                    </div>
                  )}
                </div>

                {error && (
                  <Alert className="bg-red-500/10 border-red-500/20">
                    <AlertDescription className="text-red-400">{error}</AlertDescription>
                  </Alert>
                )}

                <div className="flex gap-4 justify-center">
                  <Button
                    onClick={handleUpload}
                    disabled={!selectedImage || loading}
                    className="bg-accent-primary hover:bg-accent-primary/90 text-primary-dark px-8 py-2"
                  >
                    {loading ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-2 border-primary-dark border-t-transparent mr-2" />
                        Identifying...
                      </>
                    ) : (
                      <>
                        <Building2 className="h-4 w-4 mr-2" />
                        Identify Building
                      </>
                    )}
                  </Button>

                  {selectedImage && (
                    <Button
                      onClick={resetApp}
                      variant="outline"
                      className="border-neutral-medium/30 text-neutral-light hover:bg-neutral-medium/10"
                    >
                      Clear
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          ) : (
            /* Results Section */
            <div className="space-y-8">
              {/* Header with building name and new search button */}
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-3xl font-bold text-neutral-light mb-2">{data.building_name}</h2>
                  {data.confidence && (
                    <Badge className="bg-accent-secondary/20 text-accent-secondary">
                      {Math.round(data.confidence * 100)}% confidence
                    </Badge>
                  )}
                </div>
                <Button onClick={resetApp} className="bg-accent-primary hover:bg-accent-primary/90 text-primary-dark">
                  <Upload className="h-4 w-4 mr-2" />
                  Identify Another
                </Button>
              </div>

              <div className="grid lg:grid-cols-2 gap-8">
                {/* Images Section */}
                <div className="space-y-6">
                  {data.wikipedia_info?.image_url && (
                    <Card className="bg-neutral-dark/50 border-neutral-medium/20">
                      <CardHeader>
                        <CardTitle className="text-neutral-light"></CardTitle>
                      </CardHeader>
                      <CardContent>
                        <img
                          src={data.wikipedia_info?.image_url || "/placeholder.svg"}
                          alt="Uploaded building"
                          className="w-full rounded-lg shadow-lg"
                        />
                      </CardContent>
                    </Card>
                  )}
                </div>

                {/* Information Section */}
                <div className="space-y-6">
                  {/* Description */}
                  {data.wikipedia_info?.description && (
                    <Card className="bg-neutral-dark/50 border-neutral-medium/20">
                      <CardHeader>
                        <CardTitle className="flex items-center gap-2 text-neutral-light">
                          <Info className="h-5 w-5" />
                          About
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        <p className="text-neutral-medium leading-relaxed">{data.wikipedia_info.description}</p>
                        {data.wikipedia_info.wikipedia_link && (
                          <div className="mt-4">
                            <a
                              href={data.wikipedia_info.wikipedia_link}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-accent-primary hover:text-accent-primary/80 underline"
                            >
                              Read more on Wikipedia →
                            </a>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  )}

                  {/* Building Details Card */}
                  <Card className="bg-neutral-dark/50 border-neutral-medium/20">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-neutral-light">
                        <Building2 className="h-5 w-5" />
                        Building Details
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {data.height && (
                        <div className="flex items-center gap-3">
                          <Ruler className="h-4 w-4 text-accent-secondary" />
                          <span className="text-neutral-medium">Height:</span>
                          <span className="text-neutral-light font-medium">{data.height}</span>
                        </div>
                      )}

                      {data.floors && (
                        <div className="flex items-center gap-3">
                          <Layers className="h-4 w-4 text-accent-secondary" />
                          <span className="text-neutral-medium">Floors:</span>
                          <span className="text-neutral-light font-medium">{data.floors}</span>
                        </div>
                      )}

                      {data.status && (
                        <div className="flex items-center gap-3">
                          <span className="text-neutral-medium">Status:</span>
                          <span className="text-neutral-light font-medium">{data.status}</span>
                        </div>
                      )}

                      {data.completed && (
                        <div className="flex items-center gap-3">
                          <Calendar className="h-4 w-4 text-accent-secondary" />
                          <span className="text-neutral-medium">Completed:</span>
                          <span className="text-neutral-light font-medium">{data.completed}</span>
                        </div>
                      )}

                      {data.location && (
                        <div>
                          <span className="text-neutral-medium">Location:</span>
                          <p className="text-neutral-light font-medium mt-1">{data.location}</p>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
