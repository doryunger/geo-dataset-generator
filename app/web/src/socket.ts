import { ExtentSocket } from './api'
import { resultReceived, serverReadyReceived, store } from './store'

export const extentSocket = new ExtentSocket({
  onServerReady: () => {
    console.log('[socket] server_ready received')
    store.dispatch(serverReadyReceived())
  },
  onResult: (result) => {
    if (result.type !== 'extent_tile') {
      console.log('[socket] onResult', { type: result.type, siteCount: result.sites?.features.length })
    }
    store.dispatch(resultReceived(result))
  },
})
