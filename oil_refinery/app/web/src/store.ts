import { configureStore, createSlice, type PayloadAction } from '@reduxjs/toolkit'
import { useDispatch, useSelector, type TypedUseSelectorHook } from 'react-redux'
import { EMPTY_FEATURE_COLLECTION, INITIAL_ZOOM, type SiteFeatureCollection } from './api'

export interface Viewport {
  zoom: number
  west: number
  east: number
  south: number
  north: number
}

interface MapState {
  zoom: number
  mapLoaded: boolean
  sites: SiteFeatureCollection
  readyGeneration: number
  paintedGeneration: number
  viewport: Viewport | null
  gestureActive: boolean
  pendingFullFollowUp: boolean
}

const initialState: MapState = {
  zoom: INITIAL_ZOOM,
  mapLoaded: false,
  sites: EMPTY_FEATURE_COLLECTION,
  readyGeneration: 0,
  paintedGeneration: 0,
  viewport: null,
  gestureActive: false,
  pendingFullFollowUp: false,
}

const mapSlice = createSlice({
  name: 'map',
  initialState,
  reducers: {
    zoomChanged(state, action: PayloadAction<number>) {
      state.zoom = action.payload
    },
    mapLoaded(state) {
      state.mapLoaded = true
    },
    gestureStarted(state) {
      state.gestureActive = true
      state.pendingFullFollowUp = false
    },
    viewportSettled(state, action: PayloadAction<Viewport>) {
      state.viewport = action.payload
      state.gestureActive = false
      state.pendingFullFollowUp = true
    },
    fullFollowUpSent(state) {
      state.pendingFullFollowUp = false
    },
    extentResultReceived(state, action: PayloadAction<SiteFeatureCollection>) {
      state.sites = action.payload
      state.readyGeneration += 1
    },
    layersPainted(state, action: PayloadAction<number>) {
      state.paintedGeneration = Math.max(state.paintedGeneration, action.payload)
    },
    reset() {
      return initialState
    },
  },
})

export const {
  zoomChanged, mapLoaded, gestureStarted, viewportSettled, fullFollowUpSent,
  extentResultReceived, layersPainted, reset,
} = mapSlice.actions

export const store = configureStore({
  reducer: { map: mapSlice.reducer },
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch

export const useAppDispatch: () => AppDispatch = useDispatch
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector
