// Init page logic

async function mountInitialPage() {
  location.href = '/loading.html'
}

// Mount initial page
document.addEventListener('DOMContentLoaded', event => {
  console.log('@@ DOMContentLoaded')
  // Listen for pywebview api to be ready
  window.addEventListener('pywebviewready', async () => {
    // Mount initial page
    mountInitialPage()
  })
})
