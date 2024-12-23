import { Trophy } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'

export default function Loading() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-red-900 to-red-800">
      <header className="flex items-center justify-between p-4 bg-red-900">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-full bg-white flex items-center justify-center">
            <Trophy className="h-5 w-5 text-red-900" />
          </div>
          <span className="text-white text-lg font-bold">Dream11</span>
        </div>
      </header>

      <main className="px-4 py-6">
        <h1 className="mb-6 text-center text-2xl font-bold text-white">
          UPCOMING CRICKET MATCHES
        </h1>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[...Array(6)].map((_, index) => (
            <Card 
              key={index}
              className="bg-white/90 backdrop-blur-sm"
            >
              <CardContent className="p-4">
                <div className="mb-3 h-6 bg-gray-200 rounded animate-pulse"></div>
                <div className="mb-3 flex items-center justify-between">
                  <div className="h-4 w-20 bg-gray-200 rounded animate-pulse"></div>
                  <div className="h-6 w-24 bg-red-100 rounded-full animate-pulse"></div>
                  <div className="h-4 w-20 bg-gray-200 rounded animate-pulse"></div>
                </div>
                <div className="mb-3 flex justify-between text-center">
                  <div>
                    <div className="h-3 w-16 bg-gray-200 rounded mb-1 animate-pulse"></div>
                    <div className="h-4 w-20 bg-gray-200 rounded animate-pulse"></div>
                  </div>
                  <div>
                    <div className="h-3 w-16 bg-gray-200 rounded mb-1 animate-pulse"></div>
                    <div className="h-4 w-20 bg-gray-200 rounded animate-pulse"></div>
                  </div>
                </div>
                <div className="mb-3 flex justify-between">
                  <div className="h-3 w-24 bg-gray-200 rounded animate-pulse"></div>
                  <div className="h-3 w-16 bg-gray-200 rounded animate-pulse"></div>
                </div>
                <div className="flex items-center justify-between">
                  <div>
                    <div className="h-3 w-12 bg-gray-200 rounded mb-1 animate-pulse"></div>
                    <div className="h-4 w-16 bg-gray-200 rounded animate-pulse"></div>
                  </div>
                  <div className="h-8 w-24 bg-green-200 rounded animate-pulse"></div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </main>
    </div>
  )
}

