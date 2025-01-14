// Backend funcs
async function startServer() {
  const form = document.querySelector('form')
  // Get form data
  const formData = new FormData(form)
  const config = Object.fromEntries(formData.entries())
  config.port = parseInt(config.port)
  await window.pywebview.api.start_server_process(config)
  // Nav to Obrew Studio WebUI
  // The params help front-end know what server to connect to
  window.location = `${window.frontend.data.webui_url}/?hostname=${window.frontend.data.local_url}&port=${window.frontend.data.port}`
}
async function shutdownServer() {
  await window.pywebview.api.shutdown_server()
}

// Front-End funcs
async function getPageData() {
  const port = document.getElementById('port').value
  const data = await window.pywebview.api.update_entry_page(port)
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
    if (Object.keys(window.frontend.data).length === 0) window.frontend.data = data
    // Generate qr code
    updateQRCode(data)
    // Parse page with data
    const hostEl = document.getElementById('host')
    hostEl.value = window.frontend.data.host || data.host
    const portEl = document.getElementById('port')
    portEl.value = window.frontend.data.port || data.port
    const webuiEl = document.getElementById('webui')
    webuiEl.value = window.frontend.data.webui || data.webui_url
  } catch (error) {
    console.log('Failed to mount page', error)
  }
}
function updatePageData(ev) {
  // Update page state
  window.frontend.data[ev.target.name] = ev.target.value
  hideAdvanced()
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
}

// Global Vars
if (!window.frontend) window.frontend.data = {}
// Listeners
document.querySelector('.formOptions').addEventListener('change', updatePageData)
// Mount page
mountPage()
