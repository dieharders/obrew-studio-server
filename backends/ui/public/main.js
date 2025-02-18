// Backend funcs
async function launchWebUIFailed(err) {
  console.error('Failed to start API server')
  // Reset state if server start fails
  const btnEl = document.getElementById('startBtn')
  const settingsEl = document.getElementById('settingsBtn')
  if (btnEl) btnEl.disabled = false
  if (settingsEl) settingsEl.removeAttribute('disabled')
  // Display error message
  const toastEl = document.getElementById('messageBannerContent')
  if (toastEl) {
    toastEl.innerHTML = err
    // Set timeout to hide message
    setTimeout(() => {
      toastEl.innerHTML = ''
    }, 10000)
  }
  return '' // always return something
}
async function launchWebUI() {
  // Nav to Obrew Studio WebUI
  // The params help front-end know what server to connect to
  const target = window.frontend.state.webui || window.frontend.state.webui_url
  window.location = `${target}/?hostname=${window.frontend.state.local_url}&port=${window.frontend.state.port}`
  return '' // always return something
}
async function startServer() {
  const btnEl = document.getElementById('startBtn')
  const settingsEl = document.getElementById('settingsBtn')
  let timerRef
  try {
    const form = document.querySelector('form')
    // Disable buttons
    if (btnEl) btnEl.disabled = true
    if (settingsEl) settingsEl.setAttribute('disabled', 'disabled')
    timerRef = setTimeout(() => {
      if (btnEl) btnEl.disabled = false
      if (settingsEl) settingsEl.removeAttribute('disabled')
    }, 30000) // un-disable after 30sec
    // Get form data
    const formData = new FormData(form)
    const config = Object.fromEntries(formData.entries())
    config.port = parseInt(config.port)
    await window.pywebview.api.start_server(config)
    return
  } catch (err) {
    if (btnEl) btnEl.disabled = false
    if (settingsEl) settingsEl.removeAttribute('disabled')
    clearTimeout(timerRef)
    console.error(`Failed to start API server: ${err}`)
  }
}
async function shutdownServer() {
  await window.pywebview.api.shutdown_server()
  return
}

// Front-End funcs
async function getPageData() {
  const port = document.getElementById('port').value
  const selected_webui_url = document.getElementById('webui').value
  const data = await window.pywebview.api.update_main_page(port, selected_webui_url)
  return data
}
function updateQRCode(data) {
  const hostEl = document.getElementById('qr_link')
  hostEl.innerHTML = `${data.remote_url}:${data.port}`
  const docsLinkEl = document.querySelector('.docs-link')
  const docsLink = `${data.local_url}:${data.port}/docs`
  docsLinkEl.innerHTML = docsLink
  docsLinkEl.setAttribute('href', docsLink)
  // Show QR-Code
  const qrcodeEl = document.getElementById('qrcode')
  const qrcodeImage = data.qr_data
  if (qrcodeImage) {
    qrcodeEl.setAttribute('data-attr', 'qrcode')
    qrcodeEl.setAttribute('alt', `qr code for ${data.remote_url}:${data.port}`)
    qrcodeEl.src = `data:image/png;base64,${qrcodeImage}`
  }
}
async function mountPage() {
  try {
    // Get data from input
    const data = await getPageData()
    if (!data) return
    // Update state on first mount
    if (Object.keys(window.frontend.state).length === 0) window.frontend.state = data
    // Generate qr code
    updateQRCode(data)
    // Parse page with data
    const hostEl = document.getElementById('host')
    hostEl.value = window.frontend.state.host || data.host
    const portEl = document.getElementById('port')
    portEl.value = window.frontend.state.port || data.port
    const webuiEl = document.getElementById('webui')
    webuiEl.value = window.frontend.state.webui || data.webui_url
    return
  } catch (error) {
    console.error('Failed to mount page', error)
  }
}
function updatePageData(ev) {
  // Update page state
  window.frontend.state[ev.target.name] = ev.target.value
  hideAdvanced()
}
async function toggleConnections() {
  const containerEl = document.getElementById('connContainer')
  const isOpen = containerEl.getAttribute('data-attr') === 'open'

  if (isOpen) containerEl.setAttribute('data-attr', 'closed')
  else containerEl.setAttribute('data-attr', 'open')
  return
}
function hideAdvanced() {
  const containerEl = document.getElementById('advContainer')
  containerEl.setAttribute('data-attr', 'closed')
}
async function toggleAdvanced() {
  const containerEl = document.getElementById('advContainer')
  const isOpen = containerEl.getAttribute('data-attr') === 'open'

  if (isOpen) containerEl.setAttribute('data-attr', 'closed')
  else {
    containerEl.setAttribute('data-attr', 'open')
    // Update data
    const data = await getPageData()
    updateQRCode(data)
  }
  return
}

// Listeners
document.querySelector('.formOptions').addEventListener('change', updatePageData)
// Mount page
mountPage()
