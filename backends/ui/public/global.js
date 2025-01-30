// Global Vars
let pywebviewready = false
if (!window.frontend) window.frontend = { state: {} } // hold front-end data

// Front-End funcs
function toggleFullScreen() {
  try {
    window.pywebview.api.toggle_fullscreen()
  } catch (e) {
    console.log(`Fullscreen err: ${e}`)
  }
}
async function mountInitialPage() {
  // Fetch the new content
  const response = await fetch('/loading.html')
  const content = await response.text()
  const parser = new DOMParser()
  const newDocument = parser.parseFromString(content, 'text/html')

  // Replace the body content
  document.body.innerHTML = newDocument.body.innerHTML
  loadPageCSS(newDocument)
  loadPageJS(newDocument)
}

// Swap CSS code on page load
function loadPageCSS(newDocument) {
  // Remove existing dynamic styles
  const existingLinks = document.querySelectorAll('link[data-dynamic]')
  existingLinks.forEach(link => link.remove())
  // Add new page-specific styles
  const stylesheets = newDocument.querySelectorAll('link[rel="stylesheet"]')
  for (const stylesheet of stylesheets) {
    const newLink = document.createElement('link')
    newLink.rel = 'stylesheet'
    newLink.href = stylesheet.href
    newLink.dataset.dynamic = 'true' // Mark as dynamically loaded
    document.head.appendChild(newLink)
  }
}

// Swap JS code on page load
function loadPageJS(newDocument) {
  // Remove existing dynamic code
  const existingLinks = document.querySelectorAll('script[data-dynamic]')
  existingLinks.forEach(link => link.remove())
  // Add new page-specific code
  const scripts = newDocument.querySelectorAll('script')
  for (const script of scripts) {
    const newScript = document.createElement('script')
    newScript.dataset.dynamic = 'true' // Mark as dynamically loaded
    if (script.src) {
      newScript.src = script.src // External scripts
    } else {
      newScript.textContent = script.textContent // Inline scripts
    }
    document.head.appendChild(newScript)
  }
}

function transitionPage(url) {
  // If not supported
  if (!document.startViewTransition) {
    location.href = url
    return
  }

  document.startViewTransition(async () => {
    // Fetch the new content
    const response = await fetch(url)
    const content = await response.text()
    const parser = new DOMParser()
    const newDocument = parser.parseFromString(content, 'text/html')

    // Replace the body content
    document.body.innerHTML = newDocument.body.innerHTML

    // Dynamically load page-specific CSS
    loadPageCSS(newDocument)

    // Dynamically load page-specific javascript
    loadPageJS(newDocument)
  })
}

// Listen for fullscreen toggle keypress
document.addEventListener('keydown', event => {
  if (event.key === 'F11') toggleFullScreen()
})

// View Transition
document.addEventListener('click', e => {
  const link = e.target.closest('[data-link]')
  if (!link) return

  e.preventDefault()

  const url = link.getAttribute('href')

  transitionPage(url)
})

// Listen for pywebview api to be ready
window.addEventListener('pywebviewready', async () => {
  pywebviewready = true
  // Mount initial page
  mountInitialPage()
})
