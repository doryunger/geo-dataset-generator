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

export type BrowseMode = 'guided' | 'free'

export const FLIGHT_MS = 7000

export interface SiteProgress {
  done: number
  total: number
  prefetching: boolean
}

interface MapState {
  mode: BrowseMode
  zoom: number
  mapLoaded: boolean
  sites: SiteFeatureCollection
  detections: DetectionFeatureCollection
  graph: Graph | null
  readyGeneration: number
  paintedGeneration: number
  viewport: Viewport | null
  gestureActive: boolean
  flyTo: { bbox: [number, number, number, number]; durationMs: number; generation: number } | null
  selectedSite: Site | null
  sitePhase: SitePhase | null
  siteProgress: SiteProgress
  siteVerdicts: Record<string, boolean>
}

const initialState: MapState = {
  mode: 'guided',
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
  siteProgress: { done: 0, total: 0, prefetching: false },
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
      if (result.type === 'extent' && state.selectedSite) return
      if (result.type !== 'extent' && result.site !== state.selectedSite?.id) return
      if (result.type === 'site_start') {
        state.siteProgress = {
          done: 0, total: result.total ?? state.siteProgress.total, prefetching: false,
        }
        return
      }
      if (result.type === 'site_tile') {
        if (result.detections) state.detections.features.push(...result.detections.features)
        state.siteProgress = {
          ...state.siteProgress, done: result.done ?? 0, total: result.total ?? 0,
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
    siteSelected(state, action: PayloadAction<{ site: Site; durationMs?: number }>) {
      const { site, durationMs = FLIGHT_MS } = action.payload
      state.selectedSite = site
      state.sitePhase = 'landing'
      state.siteProgress = { done: 0, total: site.tiles, prefetching: false }
      state.graph = null
      state.sites = EMPTY_FEATURE_COLLECTION
      state.detections = EMPTY_DETECTIONS
      state.readyGeneration += 1
      state.flyTo = { bbox: site.bbox, durationMs, generation: (state.flyTo?.generation ?? 0) + 1 }
    },
    siteProcessingStarted(state) {
      state.sitePhase = 'processing'
      state.siteProgress = { ...state.siteProgress, done: 0, prefetching: true }
    },
    modeChanged(state, action: PayloadAction<BrowseMode>) {
      state.mode = action.payload
      state.selectedSite = null
      state.sitePhase = null
      state.graph = null
      state.sites = EMPTY_FEATURE_COLLECTION
      state.detections = EMPTY_DETECTIONS
      state.siteProgress = { done: 0, total: 0, prefetching: false }
      state.readyGeneration += 1
    },
    reset() {
      return initialState
    },
  },
})

export const {
  zoomChanged, mapLoaded, gestureStarted, viewportSettled,
  resultReceived, layersPainted, siteSelected, siteProcessingStarted, modeChanged, reset,
} = mapSlice.actions

interface ConnectionState {
  serverReady: boolean
  backendWarm: boolean
}

const connectionInitialState: ConnectionState = {
  serverReady: false,
  backendWarm: false,
}

const connectionSlice = createSlice({
  name: 'connection',
  initialState: connectionInitialState,
  reducers: {
    serverReadyReceived(state) {
      state.serverReady = true
    },
    backendWarmed(state) {
      state.backendWarm = true
    },
  },
})

export const { serverReadyReceived, backendWarmed } = connectionSlice.actions

export const store = configureStore({
  reducer: { map: mapSlice.reducer, connection: connectionSlice.reducer },
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch

export const useAppDispatch: () => AppDispatch = useDispatch
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector
