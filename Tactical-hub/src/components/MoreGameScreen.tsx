import { useEffect, useRef, useState } from "react";
import {
  resumeLocalGameSession,
  type ResumedLocalGameSession,
  type ResumeLocalGameSessionOptions,
} from "../game/save/localGameSession";
import type {
  LocalGameSaveMetadataRecord,
  LocalGameSaveRepository,
  LocalGameSaveRepositoryError,
  LocalGameSaveRepositoryResult,
} from "../game/save/localGameSaveStorageTypes";

type ResumeLocalGame = (
  options: ResumeLocalGameSessionOptions,
) => Promise<LocalGameSaveRepositoryResult<ResumedLocalGameSession>>;

type MoreGameScreenProps = {
  repository: LocalGameSaveRepository;
  onResume: (session: ResumedLocalGameSession) => void;
  resumeSession?: ResumeLocalGame;
  formatDateTime?: (isoTimestamp: string) => string;
};

type LoadState =
  | { status: "loading" }
  | { status: "error"; error: LocalGameSaveRepositoryError }
  | { status: "ready"; saves: LocalGameSaveMetadataRecord[] };

function defaultFormatDateTime(isoTimestamp: string) {
  const value = new Date(isoTimestamp);
  if (Number.isNaN(value.getTime())) return isoTimestamp;
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(value);
}

function controllerSummary(record: LocalGameSaveMetadataRecord) {
  const labels = {
    human: "Human",
    random_cpu: "Random CPU",
    heuristic_cpu: "Heuristic CPU",
    bc_cpu: "BC CPU",
  } as const;
  return Object.entries(record.metadata.controllersByTeamId)
    .map(([teamId, controller]) => `${teamId}: ${labels[controller]}`)
    .join(" / ");
}

export function MoreGameScreen({
  repository,
  onResume,
  resumeSession = resumeLocalGameSession,
  formatDateTime = defaultFormatDateTime,
}: MoreGameScreenProps) {
  const [loadState, setLoadState] = useState<LoadState>({ status: "loading" });
  const [resumingSaveId, setResumingSaveId] = useState<string>();
  const [resumeErrors, setResumeErrors] = useState<Record<string, LocalGameSaveRepositoryError>>({});
  const [deleteCandidateId, setDeleteCandidateId] = useState<string>();
  const [deletingSaveId, setDeletingSaveId] = useState<string>();
  const [deleteErrors, setDeleteErrors] = useState<Record<string, LocalGameSaveRepositoryError>>({});
  const resumeInFlightRef = useRef(new Set<string>());
  const deleteInFlightRef = useRef(new Set<string>());

  useEffect(() => {
    let active = true;
    setLoadState({ status: "loading" });
    void repository.list().then((result) => {
      if (!active) return;
      setLoadState(result.ok ? { status: "ready", saves: result.value } : { status: "error", error: result.error });
    });
    return () => { active = false; };
  }, [repository]);

  async function continueSave(saveId: string) {
    if (resumeInFlightRef.current.has(saveId)) return;
    resumeInFlightRef.current.add(saveId);
    setResumingSaveId(saveId);
    setResumeErrors((current) => {
      const next = { ...current };
      delete next[saveId];
      return next;
    });
    const result = await resumeSession({ repository, saveId });
    resumeInFlightRef.current.delete(saveId);
    setResumingSaveId(undefined);
    if (result.ok) {
      onResume(result.value);
      return;
    }
    setResumeErrors((current) => ({ ...current, [saveId]: result.error }));
  }

  async function confirmDelete(saveId: string) {
    if (deleteInFlightRef.current.has(saveId)) return;
    deleteInFlightRef.current.add(saveId);
    setDeletingSaveId(saveId);
    setDeleteErrors((current) => {
      const next = { ...current };
      delete next[saveId];
      return next;
    });
    const result = await repository.delete(saveId);
    deleteInFlightRef.current.delete(saveId);
    setDeletingSaveId(undefined);
    if (!result.ok) {
      setDeleteErrors((current) => ({ ...current, [saveId]: result.error }));
      return;
    }
    setDeleteCandidateId(undefined);
    setLoadState((current) => current.status === "ready"
      ? { status: "ready", saves: current.saves.filter((save) => save.saveId !== saveId) }
      : current);
  }

  if (loadState.status === "loading") return <div className="more-game-state" role="status">中断データを読み込んでいます…</div>;
  if (loadState.status === "error") return <div className="more-game-state more-game-error" role="alert" data-error-code={loadState.error.code}>
    保存データを読み込めませんでした。
  </div>;
  if (loadState.saves.length === 0) return <div className="more-game-state">中断中のLOCAL GAMEはありません。</div>;

  return <div className="more-game-list" aria-label="中断中のLOCAL GAME一覧">
    {loadState.saves.map((record) => {
      const resumeError = resumeErrors[record.saveId];
      const deleteError = deleteErrors[record.saveId];
      const isResuming = resumingSaveId === record.saveId;
      const isDeleting = deletingSaveId === record.saveId;
      return <article className="more-game-card" key={record.saveId} data-save-id={record.saveId}>
        <div className="more-game-card-heading">
          <div>
            <p className="more-game-kind">LOCAL GAME</p>
            <h2>{record.metadata.displayName}</h2>
          </div>
          <time dateTime={record.updatedAt}>{formatDateTime(record.updatedAt)}</time>
        </div>
        <dl className="more-game-details">
          <div><dt>Turn</dt><dd>{record.metadata.turnNumber}</dd></div>
          <div><dt>Phase</dt><dd>{record.metadata.phase}</dd></div>
          <div className="more-game-details-wide"><dt>CPU構成</dt><dd>{controllerSummary(record)}</dd></div>
          <div><dt>Map</dt><dd>{record.metadata.preview.mapName}</dd></div>
          <div><dt>生存ユニット</dt><dd>{record.metadata.preview.livingUnitCount}</dd></div>
        </dl>
        {resumeError ? <p className="more-game-inline-error" role="alert" data-error-code={resumeError.code}>この中断データを再開できませんでした。</p> : null}
        {deleteError ? <p className="more-game-inline-error" role="alert" data-error-code={deleteError.code}>中断データを削除できませんでした。</p> : null}
        <div className="more-game-actions">
          <button className="menu-button menu-button-primary" type="button" disabled={isResuming || isDeleting} onClick={() => { void continueSave(record.saveId); }}>
            {isResuming ? "読み込み中…" : "CONTINUE"}
          </button>
          <button className="menu-button menu-button-danger" type="button" disabled={isResuming || isDeleting} onClick={() => setDeleteCandidateId(record.saveId)}>DELETE</button>
        </div>
        {deleteCandidateId === record.saveId ? <div className="delete-confirmation" role="dialog" aria-modal="true" aria-label="中断データの削除確認">
          <p>この中断データを削除しますか？</p>
          <div className="more-game-actions">
            <button className="menu-button" type="button" disabled={isDeleting} onClick={() => setDeleteCandidateId(undefined)}>CANCEL</button>
            <button className="menu-button menu-button-danger" type="button" disabled={isDeleting} onClick={() => { void confirmDelete(record.saveId); }}>
              {isDeleting ? "削除中…" : "DELETE"}
            </button>
          </div>
        </div> : null}
      </article>;
    })}
  </div>;
}
