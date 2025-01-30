// Front-End funcs
async function downloadData() {
  const data = await window.pywebview.api.update_entry_page()
  return data
}

async function mountPage() {
  try {
    // Get data from input
    await downloadData()
    // Go to main.html page
    // eslint-disable-next-line no-undef
    transitionPage('main.html') // this exists in global.js
    return
  } catch (error) {
    console.log('Failed to mount page', error)
  }
}

// Mount page
mountPage()
