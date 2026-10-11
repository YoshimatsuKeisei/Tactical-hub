import { createContext, useContext, useEffect, useState, useSyncExternalStore, type ReactNode } from "react";
import { EMPTY_ASSETS, TileAssetStore } from "../presentation/tileAssets";
const AssetContext = createContext<TileAssetStore | null>(null);
const noopSubscribe = () => () => {};
const emptySnapshot = () => EMPTY_ASSETS;
export function TileAssetProvider({ children }: { children: ReactNode }) {
  const [store] = useState(() => new TileAssetStore());
  useEffect(() => { store.activate(); return () => store.dispose(); }, [store]);
  return <AssetContext.Provider value={store}>{children}</AssetContext.Provider>;
}
export function useTileAssets() {
  const store = useContext(AssetContext);
  const snapshot = useSyncExternalStore(store?.subscribe ?? noopSubscribe, store?.getSnapshot ?? emptySnapshot, emptySnapshot);
  return { store, snapshot };
}
