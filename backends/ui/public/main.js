// Backend funcs
function disableAllAppCards() {
  const allCards = document.querySelectorAll('.appCard')
  allCards.forEach(card => {
    card.style.pointerEvents = 'none'
    card.style.opacity = '0.6'
    // Disable all buttons within the card
    const buttons = card.querySelectorAll('button')
    buttons.forEach(btn => (btn.disabled = true))
    // Disable input within the card
    const inputs = card.querySelectorAll('input')
    inputs.forEach(input => (input.disabled = true))
  })
}

function enableAllAppCards() {
  const allCards = document.querySelectorAll('.appCard')
  allCards.forEach(card => {
    card.style.pointerEvents = 'auto'
    card.style.opacity = '1'
    // Enable all buttons within the card
    const buttons = card.querySelectorAll('button')
    buttons.forEach(btn => (btn.disabled = false))
    // Enable input within the card
    const inputs = card.querySelectorAll('input')
    inputs.forEach(input => (input.disabled = false))
  })
}

async function launchWebUIFailed(err) {
  console.error('Failed to start API server')
  // Reset state if server start fails
  const btnEl = document.getElementById('startBtn')
  const settingsEl = document.getElementById('settingsBtn')
  if (btnEl) btnEl.disabled = false
  if (settingsEl) settingsEl.removeAttribute('disabled')
  // Re-enable all app cards
  enableAllAppCards()
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
// Nav to Obrew Studio WebUI
async function launchWebUI() {
  // The params help front-end know what server to connect to
  const target = window.frontend.state.webui
  const hostEl = document.getElementById('host')
  const hostname = hostEl.value || ''
  const portEl = document.getElementById('port')
  const port = portEl.value || ''
  // Go to app
  window.location = `${target}/?hostname=${hostname}&port=${port}`
  return '' // always return something
}
async function startServer() {
  const btnEl = document.getElementById('startBtn')
  const settingsEl = document.getElementById('settingsBtn')
  let timerRef
  try {
    // Check for empty startup values
    const hostEl = document.getElementById('host')
    const hostname = hostEl.value || ''
    const portEl = document.getElementById('port')
    const port = portEl.value || ''
    if (
      !window.frontend.state.webui ||
      window.frontend.state.webui.length <= 0 ||
      !hostname ||
      !port
    ) {
      const err = 'No target, hostname or port provided to launch App.'
      launchWebUIFailed(err)
      throw new Error(err)
    }
    // Disable all app cards
    disableAllAppCards()
    const form = document.querySelector('form')
    // Disable buttons
    if (btnEl) btnEl.disabled = true
    if (settingsEl) settingsEl.setAttribute('disabled', 'disabled')
    timerRef = setTimeout(() => {
      if (btnEl) btnEl.disabled = false
      if (settingsEl) settingsEl.removeAttribute('disabled')
      enableAllAppCards()
    }, 30000) // un-disable after 30sec
    // Get form data
    const formData = new FormData(form)
    const config = Object.fromEntries(formData.entries())
    config.port = parseInt(config.port)
    // Always use the webui value from state, not from the input field
    config.webui = window.frontend.state.webui
    await window.pywebview.api.start_server(config)
    return
  } catch (err) {
    if (btnEl) btnEl.disabled = false
    if (settingsEl) settingsEl.removeAttribute('disabled')
    enableAllAppCards()
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
    // Don't set the webui input value - let it stay empty or use user input
    const currentWebui = window.frontend.state.webui || data.webui_url
    // Pre-select the app card that matches the current webui URL
    const allCards = document.querySelectorAll('.appCard')
    allCards.forEach(card => {
      if (card.getAttribute('data-webui') === currentWebui) {
        card.classList.add('selected')
      } else {
        card.classList.remove('selected')
      }
    })
    const versionEl = document.getElementById('version')
    const ver = window.frontend.state.current_version || ''
    if (ver) versionEl.innerText = `Version ${ver}`
    else versionEl.hidden = true
    // Attempt to parse update data (always do last)
    const updateMessageEl = document.getElementById('updateMessageContainer')
    const updateData = window.frontend.state.update_available || data.update_available
    updateMessageEl.hidden = !updateData
    if (!updateData) return
    const updateLinkEl = document.getElementById('updateLink')
    updateLinkEl.innerText = updateData.downloadName
    updateLinkEl.href = updateData.downloadUrl
    const updateVersionEl = document.getElementById('updateVersion')
    updateVersionEl.innerText = `New update (${updateData.tag_name}) available:`
    const updateAssetEl = document.getElementById('updateAsset')
    updateAssetEl.innerText = `Size: ${updateData.downloadSize} bytes`
    return
  } catch (error) {
    console.error('Failed to mount page', error)
    return
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
function updateCardState(el) {
  // Update selected state for all cards
  const allCards = document.querySelectorAll('.appCard')
  allCards.forEach(card => card.classList.remove('selected'))
  el.classList.add('selected')
}
function updateWebUI(el) {
  const webuiUrl = el.getAttribute('data-webui')

  // Update the webui input field
  const webuiEl = document.getElementById('webui')
  webuiEl.value = webuiUrl

  // Update page state
  window.frontend.state.webui = webuiUrl
}
// Only updates the webui input without starting the server
function selectAppCard(cardElement) {
  // Update cards
  updateCardState(cardElement)

  // Hide advanced options
  hideAdvanced()

  return
}
// Select the app and start Server
function launchApp(btnElement, event) {
  // Prevent card click event from firing
  event.stopPropagation()

  // Get the app card element (parent of parent of button)
  const appCard = btnElement.closest('.appCard')

  // Update
  selectAppCard(appCard)

  // Get the webui URL from the card's data-webui attribute
  const webuiUrl = appCard.getAttribute('data-webui')

  // Update the webui state with the card's URL
  window.frontend.state.webui = webuiUrl

  // Update cards
  updateCardState(appCard)

  // Hide advanced options
  hideAdvanced()

  // Start the server
  startServer()

  return
}
// Update webui value when custom server input changes
function updateCustomServerWebUI(event) {
  // Update page state
  window.frontend.state.webui = event.target.value
}
// Launch custom server
function launchCustomServer(event) {
  // Prevent any parent click events
  event.stopPropagation()

  // Start the server
  startServer()

  return
}

// Listeners
document.querySelector('.formOptions').addEventListener('change', updatePageData)
// Mount page
mountPage()
