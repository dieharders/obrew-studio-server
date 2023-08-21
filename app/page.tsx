'use client'

import { useState } from 'react'
// Importing {invoke, api} causes "ReferenceError: navigator is not defined" which blocks during `tauri build`
// import { open } from '@tauri-apps/api/dialog'
// import { desktopDir } from '@tauri-apps/api/path'

// import Image from "next/image";
// import Link from "next/link";

declare global {
  interface Window {
    __TAURI__: any
  }
}

const aiModelFileNames = {
  Llama13b: { fileName: 'llama-13b.ggmlv3.q3_K_S.bin', link: '' },
  Llama2_7b: { fileName: 'Llama2_7b.bin', link: '' },
  Vicuna: { fileName: 'Vicuna.bin', link: '' },
}

export default function Home() {
  const appLink = 'https://brain-dump-dieharders.vercel.app/'
  const ip = 'http://localhost:8008'
  const ITEM_MODEL_PATH = 'model-path'
  const ITEM_CURRENT_MODEL = 'current-text-model'
  const [isStarted, setIsStarted] = useState(false)
  const [modelPath, setModelPath] = useState<string>(localStorage.getItem(ITEM_MODEL_PATH) || '')
  const [currentTextModel, setCurrentTextModel] = useState<string>(
    localStorage.getItem(ITEM_CURRENT_MODEL) || '',
  )

  const onTestInference = async () => {
    console.log('@@ Testing inference...')

    const options = {
      prompt: 'Whats your name',
    }

    try {
      const response = await fetch(ip + '/v1/completions', {
        method: 'POST',
        mode: 'cors', // must be enabled otherwise wont redirect
        redirect: 'follow', // we want to follow the re-direct automatically
        cache: 'no-cache',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(options),
      })
      const result = await response.json()
      console.log('@@ [onLoadModel] Result:', result)
    } catch (error) {
      console.log('@@ [Error] Failed to connect to backend:', error)
    }
  }
  const onStart = async () => {
    console.log('@@ Starting inference...')

    const options = {
      filePath: `${modelPath}/${currentTextModel}`,
    }

    try {
      const response = await fetch(ip + '/api/text/v1/inference/start', {
        method: 'POST',
        cache: 'no-cache',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(options),
      })

      const result = await response.json()
      setIsStarted(result?.success)
      console.log('@@ [onStart] Success:', result)
    } catch (error) {
      console.log('@@ [Error] Failed to load the model:', error)
    }
  }
  const fileSelect = async (isDirMode: boolean) => {
    const { desktopDir } = window.__TAURI__.path
    const cwd = await desktopDir()
    const properties = {
      defaultPath: cwd,
      directory: isDirMode,
      filters: [
        {
          extensions: ['txt', 'gif'],
          name: '*',
        },
      ],
    }

    const { open } = window.__TAURI__.dialog
    const selected = await open(properties)
    if (Array.isArray(selected)) {
      console.log('@@ Error: user selected multiple files.')
      return null
    } else if (selected === null) {
      console.log('@@ User cancelled the selection.')
    } else {
      console.log('@@ User selected a single file:', selected)
    }
    return selected
  }

  const sizingStyles = 'lg:static lg:w-auto lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30'
  const colorStyles =
    'border-b border-gray-300 bg-gradient-to-b from-zinc-200 backdrop-blur-2xl dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit'
  /**\
   * Choose file path for ai model
   */
  const renderFilePathChooser = () => {
    return (
      <>
        {/* Path string */}
        <span
          className={`w-72 overflow-hidden text-ellipsis whitespace-nowrap pb-6 pt-8 ${colorStyles} rounded-none lg:static lg:border lg:p-4`}
          style={{ color: `${isStarted ? 'grey' : 'inherit'}` }}
        >
          {modelPath}
        </span>
        {/* Button */}
        <form className={`pb-6 pt-8 ${colorStyles} ${sizingStyles} rounded-l-none rounded-r-xl`}>
          <button
            type="button"
            id="openFileDialog"
            disabled={isStarted}
            onClick={async () => {
              const path = await fileSelect(true)
              path && setModelPath(path)
              path && localStorage.setItem(ITEM_MODEL_PATH, path)
            }}
            style={{ color: `${isStarted ? 'grey' : 'yellow'}` }}
          >
            ...
          </button>
        </form>
      </>
    )
  }
  /**
   * Start Inference Engine
   */
  const renderStartEngine = () => {
    return (
      <p className={`mr-4 rounded-xl ${colorStyles} ${sizingStyles}`}>
        <button onClick={onStart}>
          <code
            className="font-mono font-bold"
            style={{ color: `${isStarted ? 'lime' : 'yellow'}` }}
          >
            {isStarted ? '[ON]' : '[OFF]'}&nbsp;
          </code>
          <code className="font-mono font-bold">Start Engine</code>
        </button>
      </p>
    )
  }
  /**
   * Choose an ai model id
   */
  const renderModelChooser = () => {
    return (
      <p
        className={`rounded-r-none ${colorStyles} ${sizingStyles}`}
        style={{ color: `${isStarted ? 'grey' : 'inherit'}` }}
      >
        <label className="font-mono font-bold">Current model </label>
        <select
          name="modelSelect"
          id="models"
          className="bg-gray-800"
          required
          disabled={isStarted}
          value={currentTextModel}
          onChange={e => {
            const val = e?.target?.value
            console.log('@@ set curr model:', val)
            val && setCurrentTextModel(val)
            val && localStorage.setItem(ITEM_CURRENT_MODEL, val)
          }}
        >
          <option value={aiModelFileNames['Llama13b'].fileName}>Llama 13b</option>
          <option value={aiModelFileNames['Llama2_7b'].fileName}>Llama 2 7b</option>
          <option value={aiModelFileNames['Vicuna'].fileName}>Vicuna</option>
        </select>
      </p>
    )
  }
  /**
   * Download the currently selected ai model
   */
  const renderDownloadModel = () => {
    return (
      <p className={`rounded-l-xl rounded-r-none ${colorStyles} ${sizingStyles}`}>
        <button
          disabled={isStarted}
          onClick={() => {
            // Download model from huggingface
            // ...
          }}
        >
          <code
            className="font-mono font-bold"
            style={{ color: `${isStarted ? 'grey' : 'yellow'}` }}
          >
            Download
          </code>
        </button>
      </p>
    )
  }
  const renderConfigMenu = () => {
    return (
      <div
        className={`fixed left-0 top-0 flex w-full justify-center backdrop-blur-2xl dark:border-neutral-900 dark:bg-zinc-800/30 dark:from-inherit lg:rounded-xl ${sizingStyles}`}
      >
        {renderStartEngine()}
        {renderDownloadModel()}
        {renderModelChooser()}
        {renderFilePathChooser()}
      </div>
    )
  }

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex">
        {renderConfigMenu()}
        <div className="fixed bottom-0 left-0 flex h-48 w-full items-end justify-center bg-gradient-to-t from-white via-white dark:from-black dark:via-black lg:static lg:h-auto lg:w-auto lg:bg-none">
          <button
            onClick={onTestInference}
            className="pointer-events-none flex place-items-center gap-2 p-8 lg:pointer-events-auto lg:p-0"
          >
            By{' '}
            {/* <Image
              src="/vercel.svg"
              alt="Vercel Logo"
              className="dark:invert"
              width={100}
              height={24}
              priority
            /> */}
            <h2 className="text-md">Spread Shot Studios</h2>
          </button>
        </div>
      </div>

      <div className="relative flex place-items-center before:absolute before:h-[300px] before:w-[480px] before:-translate-x-1/2 before:rounded-full before:bg-gradient-radial before:from-white before:to-transparent before:blur-2xl before:content-[''] after:absolute after:-z-20 after:h-[180px] after:w-[240px] after:translate-x-1/3 after:bg-gradient-conic after:from-sky-200 after:via-blue-200 after:blur-2xl after:content-[''] before:dark:bg-gradient-to-br before:dark:from-transparent before:dark:to-blue-700 before:dark:opacity-10 after:dark:from-sky-900 after:dark:via-[#0141ff] after:dark:opacity-40 before:lg:h-[360px]">
        {/* <Image
          className="relative dark:drop-shadow-[0_0_0.3rem_#ffffff70] dark:invert"
          src="/next.svg"
          alt="Next.js Logo"
          width={180}
          height={37}
          priority
        /> */}
        <h1 className="text-4xl">🍺HomebrewAi</h1>
      </div>

      {/* Browse Apps */}
      <div className="mb-32 grid text-center lg:mb-0 lg:grid-cols-4 lg:text-left">
        <a
          href={appLink}
          className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
          target="_blank"
          rel="noopener noreferrer"
        >
          <h2 className="mb-3 text-xl font-semibold">
            Learn{' '}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h2>
          <p className="m-0 max-w-[30ch] text-sm opacity-50">
            Find in-depth information and share it. Search and analyze private data with agents.
          </p>
        </a>

        <a
          href={appLink}
          className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
          target="_blank"
          rel="noopener noreferrer"
        >
          <h2 className="mb-3 text-xl font-semibold">
            Create{' '}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h2>
          <p className="m-0 max-w-[30ch] text-sm opacity-50">
            Find inspiration, kick-off a project or just toss ideas around with a creative avatar.
          </p>
        </a>

        <a
          href={appLink}
          className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
          target="_blank"
          rel="noopener noreferrer"
        >
          <h2 className="mb-3 text-xl font-semibold">
            Heal{' '}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h2>
          <p className="m-0 max-w-[30ch] text-sm opacity-50">
            Explore your mind then reflect on your journey with an ai buddy by your side.
          </p>
        </a>

        <a
          href={appLink}
          className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
          target="_blank"
          rel="noopener noreferrer"
        >
          <h2 className="mb-3 text-xl font-semibold">
            Grow{' '}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h2>
          <p className="m-0 max-w-[30ch] text-sm opacity-50">
            Plan, adapt, enact. Take advantage of critical thinking processes to reach your goals.
          </p>
        </a>
      </div>
    </main>
  )
}
