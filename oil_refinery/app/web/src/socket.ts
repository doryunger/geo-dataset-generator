import { ExtentSocket } from './api'
import { extentResultReceived, serverReadyReceived, store } from './store'

export const extentSocket = new ExtentSocket({
  onServerReady: () => {
    console.log('[socket] server_ready received')
    store.dispatch(serverReadyReceived())
  },
  onResult: (result) => {
    console.log('[socket] onResult', { siteCount: result.features.length })
    store.dispatch(extentResultReceived(result))
  },
})
