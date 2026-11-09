import React from 'react'

/**
 * Lightweight, zero-dependency, crash-proof Markdown renderer.
 * Safely parses bold (**), italic (*), code (`), links, headings (###),
 * and lists (ordered and bulleted) directly into React virtual DOM elements.
 */

function parseInline(text) {
  if (!text) return ''

  // Regex matches:
  // 1. **bold**
  // 2. *italic* or _italic_
  // 3. `inline code`
  // 4. [link text](url)
  const regex = /(\*\*([^*]+)\*\*|\*([^*]+)\*|_([^_]+)_|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\))/g
  const elements = []
  let lastIndex = 0
  let match
  let keyIndex = 0

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      elements.push(text.substring(lastIndex, match.index))
    }

    const fullMatch = match[0]
    if (fullMatch.startsWith('**') && fullMatch.endsWith('**')) {
      // Bold
      elements.push(
        <strong key={`b-${keyIndex++}`} className="font-semibold text-slate-900">
          {match[2]}
        </strong>
      )
    } else if ((fullMatch.startsWith('*') && fullMatch.endsWith('*')) || (fullMatch.startsWith('_') && fullMatch.endsWith('_'))) {
      // Italic
      elements.push(
        <em key={`i-${keyIndex++}`} className="italic text-slate-700">
          {match[3] || match[4]}
        </em>
      )
    } else if (fullMatch.startsWith('`') && fullMatch.endsWith('`')) {
      // Inline code
      elements.push(
        <code key={`c-${keyIndex++}`} className="bg-slate-100 text-slate-800 px-1 py-0.5 rounded font-mono text-xs">
          {match[5]}
        </code>
      )
    } else if (fullMatch.startsWith('[') && fullMatch.includes('](')) {
      // Link
      elements.push(
        <a
          key={`a-${keyIndex++}`}
          href={match[7]}
          target="_blank"
          rel="noopener noreferrer"
          className="text-cyan-600 hover:text-cyan-800 underline"
        >
          {match[6]}
        </a>
      )
    } else {
      elements.push(fullMatch)
    }

    lastIndex = regex.lastIndex
  }

  if (lastIndex < text.length) {
    elements.push(text.substring(lastIndex))
  }

  return elements.length === 1 ? elements[0] : elements
}

export default function MarkdownRenderer({ content = '', className = '' }) {
  if (!content || typeof content !== 'string') {
    return null
  }

  const rawLines = content.split('\n')
  const blocks = []
  let currentList = null // { type: 'ol' | 'ul', items: [] }

  const flushList = () => {
    if (!currentList) return
    const { type, items } = currentList
    if (type === 'ol') {
      blocks.push(
        <ol key={`ol-${blocks.length}`} className="list-decimal list-inside space-y-1 my-1.5 pl-0.5">
          {items.map((it, idx) => (
            <li key={idx} className="text-slate-800 leading-relaxed">
              {parseInline(it)}
            </li>
          ))}
        </ol>
      )
    } else {
      blocks.push(
        <ul key={`ul-${blocks.length}`} className="list-disc list-inside space-y-1 my-1.5 pl-0.5">
          {items.map((it, idx) => (
            <li key={idx} className="text-slate-800 leading-relaxed">
              {parseInline(it)}
            </li>
          ))}
        </ul>
      )
    }
    currentList = null
  }

  for (let i = 0; i < rawLines.length; i++) {
    const line = rawLines[i].trim()

    if (!line) {
      flushList()
      continue
    }

    // Heading 1-3 (e.g. ### Title)
    if (/^#{1,3}\s+/.test(line)) {
      flushList()
      const title = line.replace(/^#{1,3}\s+/, '')
      blocks.push(
        <h3 key={`h-${blocks.length}`} className="text-sm font-bold text-slate-900 mt-2 mb-1">
          {parseInline(title)}
        </h3>
      )
      continue
    }

    // Ordered list item (e.g. 1. Step or 1) Step)
    const olMatch = line.match(/^(\d+)[\.\)]\s+(.*)/)
    if (olMatch) {
      if (!currentList || currentList.type !== 'ol') {
        flushList()
        currentList = { type: 'ol', items: [] }
      }
      currentList.items.push(olMatch[2])
      continue
    }

    // Unordered list item (e.g. • Item, - Item, * Item)
    const ulMatch = line.match(/^([•\-\*])\s+(.*)/)
    if (ulMatch) {
      if (!currentList || currentList.type !== 'ul') {
        flushList()
        currentList = { type: 'ul', items: [] }
      }
      currentList.items.push(ulMatch[2])
      continue
    }

    // Standard paragraph line
    flushList()
    blocks.push(
      <p key={`p-${blocks.length}`} className="my-1 text-slate-800 leading-relaxed">
        {parseInline(line)}
      </p>
    )
  }

  flushList()

  return <div className={`text-xs sm:text-sm leading-relaxed ${className}`}>{blocks}</div>
}
