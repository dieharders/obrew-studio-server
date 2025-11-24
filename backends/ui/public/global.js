// Global Vars
if (!window.frontend) window.frontend = { state: {} } // hold front-end data

// Front-End funcs
function toggleFullScreen() {
  try {
    window.pywebview.api.toggle_fullscreen()
  } catch (e) {
    console.log(`Fullscreen err: ${e}`)
  }
}

// Mount initial page
document.addEventListener('DOMContentLoaded', event => {
  // Listen for fullscreen toggle keypress
  document.addEventListener('keydown', event => {
    if (event.key === 'F11') toggleFullScreen()
  })
})
