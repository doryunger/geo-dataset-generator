import { configureStore, createSlice, type PayloadAction } from '@reduxjs/toolkit'
import { useDispatch, useSelector, type TypedUseSelectorHook } from 'react-redux'
import {
  type ComponentSummary, type DetectionFeatureCollection, EMPTY_DETECTIONS, EMPTY_FEATURE_COLLECTION,
  INITIAL_ZOOM, type ResultMessage, type Site, type SiteFeatureCollection,
} from './api'

export interface Viewport {
  zoom: number
  west: number
  east: number
  south: number
  north: number
}

export interface Graph {
  components: ComponentSummary[]
  identified: boolean
}

export type SitePhase = 'landing' | 'processing' | 'done'

interface MapState {
  zoom: number
  mapLoaded: boolean
  sites: SiteFeatureCollection
  detections: DetectionFeatureCollection
  graph: Graph | null
  readyGeneration: number
  paintedGeneration: number
  viewport: Viewport | null
  gestureActive: boolean
  flyTo: { bbox: [number, number, number, number]; generation: number } | null
  selectedSite: Site | null
  sitePhase: SitePhase | null
  siteProgress: { done: number; total: number; startedAt: number; updatedAt: number }
  siteVerdicts: Record<string, boolean>
}

const initialState: MapState = {
  zoom: INITIAL_ZOOM,
  mapLoaded: false,
  sites: EMPTY_FEATURE_COLLECTION,
  detections: EMPTY_DETECTIONS,
  graph: null,
  readyGeneration: 0,
  paintedGeneration: 0,
  viewport: null,
  gestureActive: false,
  flyTo: null,
  selectedSite: null,
  sitePhase: null,
  siteProgress: { done: 0, total: 0, startedAt: 0, updatedAt: 0 },
  siteVerdicts: {},
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
    },
    viewportSettled(state, action: PayloadAction<Viewport>) {
      state.viewport = action.payload
      state.gestureActive = false
    },
    resultReceived(state, action: PayloadAction<ResultMessage>) {
      const result = action.payload
      if (result.type === 'extent_tile') {
        if (!result.detections) return
        state.detections.features = state.detections.features
          .filter((f) => f.properties.tile !== result.tile)
          .concat(result.detections.features)
        state.readyGeneration += 1
        return
      }
      if (result.type !== 'extent' && result.site !== state.selectedSite?.id) return
      if (result.type === 'site_tile') {
        if (result.detections) state.detections.features.push(...result.detections.features)
        state.siteProgress = {
          ...state.siteProgress, done: result.done ?? 0, total: result.total ?? 0, updatedAt: Date.now(),
        }
      } else if (result.detections) {
        state.detections = result.detections
      }
      if (result.sites) state.sites = result.sites
      state.graph = {
        components: result.components ?? state.graph?.components ?? [],
        identified: result.sites ? result.sites.features.length > 0 : (state.graph?.identified ?? false),
      }
      state.readyGeneration += 1
      if (result.type === 'site_done') {
        state.sitePhase = 'done'
        if (result.site) state.siteVerdicts[result.site] = (result.sites?.features.length ?? 0) > 0
      }
    },
    layersPainted(state, action: PayloadAction<number>) {
      state.paintedGeneration = Math.max(state.paintedGeneration, action.payload)
    },
    siteSelected(state, action: PayloadAction<Site>) {
      state.selectedSite = action.payload
      state.sitePhase = 'landing'
      state.siteProgress = { done: 0, total: action.payload.tiles, startedAt: 0, updatedAt: 0 }
      state.graph = null
      state.sites = EMPTY_FEATURE_COLLECTION
      state.detections = EMPTY_DETECTIONS
      state.readyGeneration += 1
      state.flyTo = { bbox: action.payload.bbox, generation: (state.flyTo?.generation ?? 0) + 1 }
    },
    siteProcessingStarted(state) {
      state.sitePhase = 'processing'
      state.siteProgress = { ...state.siteProgress, startedAt: Date.now(), updatedAt: Date.now() }
    },
    siteCleared(state) {
      state.selectedSite = null
      state.sitePhase = null
      state.graph = null
    },
    reset() {
      return initialState
    },
  },
})

export const {
  zoomChanged, mapLoaded, gestureStarted, viewportSettled,
  resultReceived, layersPainted, siteSelected, siteProcessingStarted, siteCleared, reset,
} = mapSlice.actions

interface ConnectionState {
  serverReady: boolean
}

const connectionInitialState: ConnectionState = {
  serverReady: false,
}

const connectionSlice = createSlice({
  name: 'connection',
  initialState: connectionInitialState,
  reducers: {
    serverReadyReceived(state) {
      state.serverReady = true
    },
  },
})

export const { serverReadyReceived } = connectionSlice.actions

export const store = configureStore({
  reducer: { map: mapSlice.reducer, connection: connectionSlice.reducer },
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch

export const useAppDispatch: () => AppDispatch = useDispatch
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector
