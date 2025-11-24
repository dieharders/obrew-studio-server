/* eslint-disable no-undef */

// Front-End funcs
async function downloadData() {
  try {
    const data = await window.pywebview.api.update_entry_page()
    return data
  } catch (error) {
    console.error(error)
    throw error
  }
}
// https://docs.github.com/en/rest/releases/releases?apiVersion=2022-11-28#get-the-latest-release
async function fetchLatestVersion() {
  try {
    const apiUrl = 'https://studio.openbrew.ai/api/github'
    // const apiUrl = 'http://localhost:3000/api/github' // test locally
    const response = await fetch(apiUrl, { method: 'GET' })
    if (!response.ok) {
      console.error(`Error while checking latest version. Status: ${response.status}`)
    } else {
      const result = await response.json()
      const data = result.data
      return data
    }
    return
  } catch (error) {
    console.error('Error while checking latest version: ', error)
    return
  }
}

async function mountPage() {
  try {
    // Get data from input
    await downloadData()

    // Initialize frontend state if needed
    if (!window.frontend.state) {
      window.frontend.state = {}
    }

    // Check for latest version on server side
    const verRes = await fetchLatestVersion()
    if (verRes) {
      const latest_tag = verRes.tag_name
      const isAvailable = await window.pywebview.api.check_is_latest_version(latest_tag)
      window.frontend.state.update_available = isAvailable ? verRes : null
    }
    // Go to main.html page
    location.href = '/main.html'
    return
  } catch (error) {
    const msg = `Error while mounting page: ${error}`
    // alert(msg)
    console.error(msg)
    // Go to main.html page
    location.href = '/main.html'
    return
  }
}

// Mount initial page
document.addEventListener('DOMContentLoaded', event => {
  // Listen for pywebview api to be ready
  window.addEventListener('pywebviewready', async () => {
    // Mount page
    mountPage()
  })
})
