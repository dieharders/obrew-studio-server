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
async function checkLatestVersion() {
  try {
    const apiUrl = 'https://api.github.com/repos/dieharders/obrew-studio-server/releases/latest'
    const githubToken = ''
    const response = await fetch(apiUrl, {
      method: 'GET',
      headers: {
        Accept: 'application/vnd.github+json',
        Authorization: `Bearer ${githubToken}`,
        'X-GitHub-Api-Version': '2022-11-28',
      },
    })
    if (!response.ok) {
      console.error(`Error while checking latest version: ${response.status}`)
    } else {
      const json = await response.json()
      return json
    }
    return
  } catch (error) {
    console.error(error)
  }
}

async function mountPage() {
  try {
    // Get data from input
    await downloadData()
    // Check for latest version
    // @TODO Perform on server side so we can pass github token for auth
    // await checkLatestVersion()
    // Go to main.html page
    // eslint-disable-next-line no-undef
    transitionPage('main.html') // this exists in global.js
    return
  } catch (error) {
    const msg = 'Failed to mount page. '
    console.error(msg, error)
    alert(msg + error)
  }
}

// Mount page
mountPage()
