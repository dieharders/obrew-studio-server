// ==================== STATE MANAGEMENT ====================
let serverStartTime = null
let uptimeInterval = null

// ==================== MODAL FUNCTIONS ====================
// Store the element that triggered the modal for focus restoration
let previousActiveElement = null

function openSettingsModal() {
  // Store current active element to restore focus later
  previousActiveElement = document.activeElement

  const modal = document.getElementById('settingsModal')
  modal.classList.add('active')
  document.body.style.overflow = 'hidden' // Prevent body scroll

  // Load current settings
  loadSettings()

  // Add keyboard listener for Escape key
  document.addEventListener('keydown', handleSettingsModalEscape)
}

function closeSettingsModal() {
  const modal = document.getElementById('settingsModal')
  modal.classList.remove('active')
  document.body.style.overflow = '' // Restore body scroll

  // Auto-save settings when modal closes
  saveSettings()

  // Remove keyboard listener
  document.removeEventListener('keydown', handleSettingsModalEscape)

  // Restore focus to previous element
  if (previousActiveElement) {
    previousActiveElement.focus()
    previousActiveElement = null
  }
}

function closeSettingsModalOnBackdrop(event) {
  if (event.target === event.currentTarget) {
    closeSettingsModal()
  }
}

function handleSettingsModalEscape(event) {
  if (event.key === 'Escape') {
    closeSettingsModal()
  }
}

async function loadSettings() {
  try {
    const data = await window.pywebview.api.update_settings_page()

    // Load settings into form fields
    const llamaCloudEl = document.getElementById('llamaCloud')
    if (data.llamaIndexAPIKey) llamaCloudEl.value = data.llamaIndexAPIKey

    const sslEl = document.getElementById('ssl')
    if (data.ssl) sslEl.checked = true
    else sslEl.checked = false

    const corsEl = document.getElementById('cors')
    if (data.cors) corsEl.value = data.cors

    const adminEl = document.getElementById('admin')
    if (data.adminWhitelist) adminEl.value = data.adminWhitelist
  } catch (error) {
    console.error('Failed to load settings:', error)
  }
}

async function saveSettings() {
  try {
    const form = document.getElementById('settingsForm')
    const formData = new FormData(form)

    // Handle checkboxes
    const checkboxes = form.querySelectorAll('input[type="checkbox"]')
    const checkData = {}
    checkboxes.forEach(checkbox => {
      checkData[checkbox.name] = checkbox.checked ? 'true' : 'false'
    })

    // Convert to object and merge
    const formDataObject = Object.fromEntries(formData.entries())
    await window.pywebview.api.save_settings({ ...formDataObject, ...checkData })
  } catch (error) {
    console.error('Failed to save settings:', error)
  }
}

// ==================== SERVER FUNCTIONS ====================

async function handleWebUILaunchError(err) {
  console.error('Failed to start API server:', err)

  // Reset state
  const btnEl = document.getElementById('startServerBtn')
  if (btnEl) btnEl.disabled = false

  // Display error message
  const toastEl = document.getElementById('messageBannerContent')
  if (toastEl) {
    toastEl.innerHTML = err
    setTimeout(() => {
      toastEl.innerHTML = ''
    }, 10000)
  }

  return ''
}

async function navigateToWebUI() {
  try {
    const target = window.frontend.state.webui
    const hostEl = document.getElementById('host')
    const hostname = hostEl.value || ''
    const portEl = document.getElementById('port')
    const port = portEl.value || ''

    if (!target) {
      throw new Error('No WebUI target specified')
    }

    // Determine protocol based on SSL setting
    const sslEnabled = await window.pywebview.api.get_ssl_setting()
    const protocol = sslEnabled ? 'https' : 'http'
    const serverUrl = `${protocol}://localhost:${port}`

    // Navigate to app
    window.location = `${target}/?protocol=${protocol}&hostname=${hostname}&port=${port}&serverUrl=${encodeURIComponent(
      serverUrl,
    )}`

    return ''
  } catch (error) {
    console.error('Failed to navigate to WebUI:', error)
    showToast(`Failed to launch app: ${error.message}`)
    throw error
  }
}

// Common function to start server with given configuration
async function startServerWithConfig(webuiTarget = '', apiMethod = 'start_headless_server') {
  const btnEl = document.getElementById('startServerBtn')
  let timerRef

  try {
    // Get connection details
    const hostEl = document.getElementById('host')
    const hostname = hostEl.value || ''
    const portEl = document.getElementById('port')
    const port = portEl.value || ''

    // Validate inputs based on mode
    if (webuiTarget && !window.frontend.state.webui) {
      const err = 'No target provided to launch App.'
      handleWebUILaunchError(err)
      throw new Error(err)
    }

    if (!hostname || !port) {
      const err = 'No hostname or port provided.'
      handleWebUILaunchError(err)
      throw new Error(err)
    }

    // Disable button
    if (btnEl) btnEl.disabled = true

    // Safety timeout to re-enable button after 30 seconds
    timerRef = setTimeout(() => {
      if (btnEl) btnEl.disabled = false
    }, 30000)

    // Get form data
    const form = document.querySelector('.connection-form')
    const formData = new FormData(form)
    const config = Object.fromEntries(formData.entries())
    config.port = parseInt(config.port)
    config.webui = webuiTarget

    // Start server with appropriate API method
    await window.pywebview.api[apiMethod](config)

    // Switch to post-server view
    await showPostServerView()

    // Clear timeout on success
    if (timerRef) clearTimeout(timerRef)

    return
  } catch (err) {
    // Clear timeout on error to prevent race condition
    if (timerRef) clearTimeout(timerRef)
    if (btnEl) btnEl.disabled = false
    console.error(`Failed to start API server: ${err}`)
    handleWebUILaunchError(err.toString())
  }
}

// Start server only (without launching an app)
async function startServerOnly() {
  try {
    return await startServerWithConfig('', 'start_headless_server')
  } catch (error) {
    console.error('Failed to start server:', error)
    showToast(`Failed to start server: ${error.message}`)
    throw error
  }
}

async function handleShutdownServer() {
  try {
    await shutdownServer()
    // Switch back to pre-server view
    showPreServerView()
    return
  } catch (err) {
    console.error('Failed to shutdown server:', err)
  }
}

async function shutdownServer() {
  try {
    await window.pywebview.api.shutdown_server()
    return
  } catch (error) {
    console.error('Failed to shutdown server:', error)
    showToast(`Failed to shutdown server: ${error.message}`)
    throw error
  }
}

// ==================== UPTIME MANAGEMENT ====================
function startUptimeCounter() {
  // Clear any existing interval to prevent memory leaks
  stopUptimeCounter()

  // Start new counter
  serverStartTime = Date.now()
  uptimeInterval = setInterval(updateUptime, 1000)
}

function stopUptimeCounter() {
  if (uptimeInterval) {
    clearInterval(uptimeInterval)
    uptimeInterval = null
  }
  serverStartTime = null
}

// ==================== VIEW SWITCHING ====================
async function showPostServerView() {
  // Hide pre-server view
  const preServerView = document.getElementById('preServerView')
  preServerView.hidden = true

  // Show post-server view
  const postServerView = document.getElementById('postServerView')
  postServerView.hidden = false

  // Update server metrics
  updateServerMetrics()

  // Start uptime counter (handles cleanup of any existing interval)
  startUptimeCounter()

  // Create app cards from hosted_apps data
  const data = await getPageData()

  if (data.hosted_apps && data.hosted_apps.length > 0) {
    createAppCards(data.hosted_apps)
  }

  // Update docs link with current server info
  const docsLinkEl = document.getElementById('docs-link')
  if (docsLinkEl && data.local_url && data.port) {
    const docsLink = `${data.local_url}:${data.port}/docs`
    docsLinkEl.innerHTML = docsLink
    docsLinkEl.setAttribute('href', docsLink)
  }

  // Set value for server status
  const sslEnabled = await window.pywebview.api.get_ssl_setting()
  const sslEl = document.getElementById('serverStatus')
  if (sslEl) {
    sslEl.textContent = `${sslEnabled ? 'Running (SSL)' : 'Running'}`
  }
}

function showPreServerView() {
  // Show pre-server view
  const preServerView = document.getElementById('preServerView')
  preServerView.hidden = false

  // Hide post-server view
  const postServerView = document.getElementById('postServerView')
  postServerView.hidden = true

  // Stop uptime counter (handles cleanup)
  stopUptimeCounter()

  // Re-enable controls
  const btnEl = document.getElementById('startServerBtn')
  if (btnEl) btnEl.disabled = false
}

// ==================== METRICS UPDATES ====================
function updateServerMetrics() {
  const hostEl = document.getElementById('host')
  const portEl = document.getElementById('port')

  const hostname = hostEl.value ? hostEl.value : 'localhost'
  const port = portEl.value ? portEl.value : '8008'

  // Update server address
  const serverAddressEl = document.getElementById('serverAddress')
  if (serverAddressEl) {
    serverAddressEl.textContent = `http://${hostname}:${port}`
  }

  // Update status
  const serverStatusEl = document.getElementById('serverStatus')
  if (serverStatusEl) {
    serverStatusEl.textContent = 'Running'
  }

  // Reset active connections
  const activeConnectionsEl = document.getElementById('activeConnections')
  if (activeConnectionsEl) {
    // @TODO Implement fetching this value from backend
    activeConnectionsEl.textContent = '1'
  }
}

function updateUptime() {
  if (!serverStartTime) return

  const uptimeEl = document.getElementById('serverUptime')
  if (!uptimeEl) return

  const now = Date.now()
  const diff = now - serverStartTime

  const hours = Math.floor(diff / (1000 * 60 * 60))
  const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60))
  const seconds = Math.floor((diff % (1000 * 60)) / 1000)

  uptimeEl.textContent = `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(
    2,
    '0',
  )}`
}

// ==================== PAGE SETUP ====================
async function getPageData() {
  const portEl = document.getElementById('port')
  const port = portEl.value ? portEl.value : ''
  const customWebUIEl = document.getElementById('customWebUI')
  const selected_webui_url = customWebUIEl.value ? customWebUIEl.value : ''

  const data = await window.pywebview.api.update_main_page(port, selected_webui_url)
  return data
}

function openQRModal() {
  // Store current active element to restore focus later
  if (!previousActiveElement) {
    previousActiveElement = document.activeElement
  }

  const modal = document.getElementById('qrModal')
  if (modal) {
    modal.classList.add('active')
    document.body.style.overflow = 'hidden' // Prevent body scroll

    // Add keyboard listener for Escape key
    document.addEventListener('keydown', handleQRModalEscape)
  }
}

function closeQRModal() {
  const modal = document.getElementById('qrModal')
  if (modal) {
    modal.classList.remove('active')
    document.body.style.overflow = '' // Restore body scroll

    // Remove keyboard listener
    document.removeEventListener('keydown', handleQRModalEscape)

    // Restore focus to previous element
    if (previousActiveElement) {
      previousActiveElement.focus()
      previousActiveElement = null
    }
  }
}

function handleQRModalEscape(event) {
  if (event.key === 'Escape') {
    closeQRModal()
  }
}

function openAppQRModal(event, element) {
  event.stopPropagation()

  // Read data from element's data attributes
  const appTitle = element.getAttribute('data-app-title')
  const qrData = element.getAttribute('data-qr-data')
  const qrUrl = element.getAttribute('data-qr-url')

  // Store current active element to restore focus later
  if (!previousActiveElement) {
    previousActiveElement = document.activeElement
  }

  // Update modal content with app-specific data
  const modalTitle = document.getElementById('qrModalTitle')
  const modalDescription = document.getElementById('qrModalDescription')
  const qrcodeEl = document.getElementById('qrcode')
  const qrLinkEl = document.getElementById('qrLink')

  if (modalTitle) {
    modalTitle.textContent = `Connect to ${appTitle}`
  }
  if (modalDescription) {
    modalDescription.textContent = `Scan this QR code to open ${appTitle} on your mobile device`
  }
  if (qrcodeEl) {
    qrcodeEl.setAttribute('data-state', 'qrcode')
    qrcodeEl.src = `data:image/png;base64,${qrData}`
    qrcodeEl.alt = `QR Code for ${appTitle}`
  }
  if (qrLinkEl) {
    qrLinkEl.textContent = qrUrl
    qrLinkEl.setAttribute('data-url', qrUrl)
  }

  // Open the modal
  const modal = document.getElementById('qrModal')
  if (modal) {
    modal.classList.add('active')
    document.body.style.overflow = 'hidden'
    document.addEventListener('keydown', handleQRModalEscape)
  }
}

function createAppCards(hostedApps) {
  if (!hostedApps || hostedApps.length === 0) return

  const appCardsContainer = document.getElementById('appCardsGrid')
  if (!appCardsContainer) return

  const customServerCard = appCardsContainer.querySelector('.app-card[data-card-type="custom"]')

  // Remove all existing app cards except the custom server card
  const allCards = appCardsContainer.querySelectorAll('.app-card')
  allCards.forEach(card => {
    if (card !== customServerCard) {
      card.remove()
    }
  })

  // Create app cards for each hosted app
  hostedApps.forEach(app => {
    const card = document.createElement('div')
    card.className = 'app-card'
    card.setAttribute('data-webui', app.url)
    card.setAttribute('data-name', app.title)
    card.onclick = function () {
      selectAppCard(this)
    }

    // Build QR code section if available
    const qrSection = app.qr_data
      ? `
      <div class="app-qr-section">
        <img
          class="app-qr-code"
          src="data:image/png;base64,${app.qr_data}"
          alt="QR Code for ${app.title}"
          title="Click to enlarge"
          data-app-title="${app.title}"
          data-qr-data="${app.qr_data}"
          data-qr-url="${app.qr_url}"
          onclick="openAppQRModal(event, this)"
        />
        <p class="app-qr-label">Scan to connect</p>
      </div>
    `
      : ''

    card.innerHTML = `
      <div class="app-card-content">
        <h3>${app.title}</h3>
        <p class="app-description">${app.description}</p>
        <div class="app-card-actions">
          ${qrSection}
          <div class="app-buttons">
            <button type="button" class="btn btn-secondary btn-app" onclick="openInBrowser(this, event)">
              Launch in Browser
            </button>
            <button type="button" class="btn btn-secondary btn-app" onclick="launchSelectedApp(this, event)">
              Open
            </button>
          </div>
        </div>
      </div>
    `

    // Insert before the custom server card
    appCardsContainer.insertBefore(card, customServerCard)
  })
}

async function mountPage() {
  try {
    // Get data from backend
    const data = await getPageData()
    if (!data) return

    // Update state on first mount
    if (Object.keys(window.frontend.state).length === 0) {
      window.frontend.state = data
    }

    // Set current version state
    const currVer = await window.pywebview.api.get_current_version()
    window.frontend.state.current_version = currVer || ''

    // Parse page with data
    const hostEl = document.getElementById('host')
    if (hostEl) {
      hostEl.value = window.frontend.state.host || data.host
    }

    const portEl = document.getElementById('port')
    if (portEl) {
      portEl.value = window.frontend.state.port || data.port
    }

    // Update version
    const versionEl = document.getElementById('version')
    const ver = window.frontend.state.current_version || ''
    if (versionEl) {
      if (ver) versionEl.innerText = `Version ${ver}`
      else versionEl.hidden = true
    }

    // Handle update notification toast
    const updateData = window.frontend.state.update_available || data.update_available

    if (updateData) {
      const updateLinkEl = document.getElementById('updateLink')
      if (updateLinkEl) {
        updateLinkEl.innerText = updateData.downloadName
        updateLinkEl.href = updateData.downloadUrl
      }

      const updateVersionEl = document.getElementById('updateVersion')
      if (updateVersionEl) {
        updateVersionEl.innerText = `New update (${updateData.tag_name}) available:`
      }

      const updateAssetEl = document.getElementById('updateAsset')
      if (updateAssetEl) {
        // Convert bytes to megabytes (MB)
        const sizeInMB = (updateData.downloadSize / (1024 * 1024)).toFixed(2)
        updateAssetEl.innerText = `Size: ${sizeInMB} MB`
      }

      // Show the update toast
      showUpdateToast()
    }

    return
  } catch (error) {
    console.error('Failed to mount page:', error)
    return
  }
}

function updatePageData(ev) {
  window.frontend.state[ev.target.name] = ev.target.value
}

// ==================== APP CARD SELECTION ====================
function updateCardState(el) {
  const allCards = document.querySelectorAll('.app-card')
  allCards.forEach(card => card.classList.remove('selected'))
  el.classList.add('selected')
}

function selectAppCard(cardElement) {
  updateCardState(cardElement)

  // Get the webui URL from the card's data-webui attribute
  const webuiUrl = cardElement.getAttribute('data-webui')
  if (webuiUrl) {
    window.frontend.state.webui = webuiUrl
  }
}

function launchSelectedApp(btnElement, event) {
  event.stopPropagation()

  const appCard = btnElement.closest('.app-card')
  selectAppCard(appCard)

  const webuiUrl = appCard.getAttribute('data-webui')
  if (webuiUrl) {
    window.frontend.state.webui = webuiUrl
    // Navigate directly without starting server (server should already be running)
    navigateToWebUI()
  }
}

function updateCustomServerWebUI(event) {
  window.frontend.state.webui = event.target.value
}

function launchCustomServer(event) {
  event.stopPropagation()

  // Get the custom WebUI URL from the input
  const customWebUIEl = document.getElementById('customWebUI')
  const webuiUrl = customWebUIEl.value

  if (webuiUrl) {
    window.frontend.state.webui = webuiUrl
    // Navigate directly without starting server (server should already be running)
    navigateToWebUI()
  }
}

async function openInBrowser(element, event) {
  event.stopPropagation()

  try {
    // Get URL from the element's data attribute or from parent card
    let url = element.getAttribute('data-url')

    if (!url) {
      const appCard = element.closest('.app-card')
      url = appCard.getAttribute('data-webui')
    }

    if (!url) {
      throw new Error('No URL specified for app')
    }

    // Get connection details
    const hostEl = document.getElementById('host')
    const hostname = hostEl.value || 'localhost'
    const portEl = document.getElementById('port')
    const port = portEl.value || '8008'

    // Determine protocol based on SSL setting
    const sslEnabled = await window.pywebview.api.get_ssl_setting()
    const protocol = sslEnabled ? 'https' : 'http'
    const serverUrl = `${protocol}://localhost:${port}`

    // Append URL parameters
    const urlWithParams = `${url}?protocol=${protocol}&hostname=${hostname}&port=${port}&serverUrl=${encodeURIComponent(
      serverUrl,
    )}`

    await window.pywebview.api.open_url_in_browser(urlWithParams)
  } catch (error) {
    console.error('Failed to open URL in browser:', error)
    showToast(`Failed to open app in browser: ${error.message}`)
  }
}

async function openQRLinkInBrowser() {
  try {
    const qrLinkEl = document.getElementById('qrLink')
    // Use data-url attribute (set by openAppQRModal) or fallback to textContent
    const url = qrLinkEl.getAttribute('data-url') || qrLinkEl.textContent

    if (!url) {
      throw new Error('No QR link URL available')
    }

    // Add protocol if not present
    const fullUrl = url.startsWith('http') ? url : `http://${url}`
    await window.pywebview.api.open_url_in_browser(fullUrl)
  } catch (error) {
    console.error('Failed to open QR link in browser:', error)
    showToast(`Failed to open link: ${error.message}`)
  }
}

// ==================== EVENT LISTENERS ====================
const connectionForm = document.querySelector('.connection-form')
if (connectionForm) {
  connectionForm.addEventListener('change', updatePageData)
}

// ==================== TOAST NOTIFICATION ====================
function dismissToast() {
  const toast = document.getElementById('messageToast')
  if (toast) {
    toast.classList.remove('show')
    // Clear the message after animation completes
    setTimeout(() => {
      const content = document.getElementById('messageBannerContent')
      if (content) content.textContent = ''
    }, 300)
  }
}

function showToast(message) {
  const toast = document.getElementById('messageToast')
  const content = document.getElementById('messageBannerContent')

  if (toast && content && message) {
    content.textContent = message
    // Small delay to ensure animation triggers
    setTimeout(() => {
      toast.classList.add('show')
    }, 10)
  }
}

function dismissUpdateToast() {
  const toast = document.getElementById('updateToast')
  if (toast) {
    toast.classList.remove('show')
  }
}

function showUpdateToast() {
  const toast = document.getElementById('updateToast')
  if (toast) {
    // Small delay to ensure animation triggers
    setTimeout(() => {
      toast.classList.add('show')
    }, 10)
  }
}

// ==================== INITIALIZE ====================

document.addEventListener('DOMContentLoaded', event => {
  // Listen for pywebview api to be ready
  window.addEventListener('pywebviewready', async () => {
    // Mount page
    mountPage()
  })
})
