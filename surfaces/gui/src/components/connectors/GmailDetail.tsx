import { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import {
  disconnectGmailAccount,
  setGmailDefaultAccount,
  setGmailFilters,
  getGmailOAuthStatus,
  configureGmailOAuth,
  signInGmailOAuth,
  type GmailAccount,
  type GmailOAuthStatus,
} from "../../api";
import { ConnectorBadge } from "../../connectors/ConnectorIcon";
import type { DetailProps } from "./ConnectorsSection";
import { ToolsDisclosure } from "./ToolsDisclosure";
import { FOOT, GRP, GRP_H, PILL_ACCENT, ROW, TAG_ACCENT, TAG_WARN, XBTN } from "./ui";

// The Gmail detail page: local OAuth flow (no cloud broker needed).
// - If Google OAuth client is not configured: show setup form
// - If configured: show "Sign in with Google" button
// - After sign-in: show connected accounts with filters

const LABEL = "text-[13px] text-muted w-24 shrink-0";

export function GmailDetail({ c, cloud: _cloud, slack: _slack, onChanged }: DetailProps) {
  const { t } = useTranslation();
  const [oauthStatus, setOauthStatus] = useState<GmailOAuthStatus | null>(null);
  const [showConfigForm, setShowConfigForm] = useState(false);
  const [clientId, setClientId] = useState("");
  const [clientSecret, setClientSecret] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const accounts = (c.accounts ?? []) as GmailAccount[];

  useEffect(() => {
    loadStatus();
  }, []);

  const loadStatus = async () => {
    try {
      const status = await getGmailOAuthStatus();
      setOauthStatus(status);
    } catch {
      // Ignore errors
    }
  };

  const handleConfigure = async () => {
    if (!clientId.trim() || !clientSecret.trim()) {
      setError(t("gmail.oauth_config_required"));
      return;
    }
    setBusy(true);
    setError(null);
    const result = await configureGmailOAuth(clientId.trim(), clientSecret.trim());
    if (result.ok) {
      setShowConfigForm(false);
      setClientId("");
      setClientSecret("");
      await loadStatus();
    } else {
      setError(result.error || t("gmail.oauth_config_failed"));
    }
    setBusy(false);
  };

  const handleSignIn = async () => {
    setBusy(true);
    setError(null);
    const result = await signInGmailOAuth();
    if (result.ok) {
      await loadStatus();
      await onChanged();
    } else {
      setError(result.error || t("gmail.oauth_signin_failed"));
    }
    setBusy(false);
  };

  const clientConfigured = oauthStatus?.client_configured ?? false;

  return (
    <div data-testid="gmail-detail">
      <div className="flex items-center gap-3.5 mb-5">
        <ConnectorBadge connector={c} size={44} title="Gmail" />
        <div className="min-w-0 flex-1">
          <h2 className="text-[20px] font-semibold tracking-tight leading-tight">Gmail</h2>
          <div className="text-[13px] text-muted flex items-center gap-1.5">
            {accounts.length > 0 ? (
              <>
                <span className="w-2 h-2 rounded-full bg-ok" />
                <span data-testid="gmail-status">
                  {t("connector.account_count", { count: accounts.length })}
                </span>
              </>
            ) : (
              <span>{t("connector.not_connected")}</span>
            )}
          </div>
        </div>
        {clientConfigured && (
          <button
            className={PILL_ACCENT}
            data-testid="add-account-btn"
            onClick={handleSignIn}
            disabled={busy}
          >
            {busy ? t("gmail.signing_in") : t("gmail.add_account")}
          </button>
        )}
      </div>

      {!clientConfigured && !showConfigForm && (
        <div className={GRP}>
          <div className={ROW + " text-[13px] text-muted"}>
            {t("gmail.local_oauth_setup_blurb")}
          </div>
          <button
            className={PILL_ACCENT + " mt-2"}
            onClick={() => setShowConfigForm(true)}
          >
            {t("gmail.configure_google_oauth")}
          </button>
        </div>
      )}

      {showConfigForm && (
        <div className={GRP}>
          <div className={ROW + " text-[13px] text-muted mb-2"}>
            {t("gmail.oauth_instructions")}
          </div>
          <ol className="text-[12px] text-muted list-decimal list-inside space-y-1 mb-3">
            <li>{t("gmail.oauth_step_1")}</li>
            <li>{t("gmail.oauth_step_2")}</li>
            <li>{t("gmail.oauth_step_3")}</li>
            <li>{t("gmail.oauth_step_4")}</li>
          </ol>
          <div className={ROW}>
            <span className={LABEL}>{t("gmail.client_id")}</span>
            <input
              type="text"
              className="flex-1 min-w-0 bg-paper border border-line rounded px-2 py-1 text-[13px] outline-none focus:border-accent"
              placeholder="xxxx.apps.googleusercontent.com"
              value={clientId}
              onChange={(e) => setClientId(e.target.value)}
            />
          </div>
          <div className={ROW}>
            <span className={LABEL}>{t("gmail.client_secret")}</span>
            <input
              type="password"
              className="flex-1 min-w-0 bg-paper border border-line rounded px-2 py-1 text-[13px] outline-none focus:border-accent"
              placeholder="GOCSPX-xxxx"
              value={clientSecret}
              onChange={(e) => setClientSecret(e.target.value)}
            />
          </div>
          <div className="flex items-center gap-2 mt-2">
            <button
              className={PILL_ACCENT}
              onClick={handleConfigure}
              disabled={busy}
            >
              {t("gmail.save_config")}
            </button>
            <button
              className="text-[12px] text-muted hover:text-ink"
              onClick={() => {
                setShowConfigForm(false);
                setClientId("");
                setClientSecret("");
                setError(null);
              }}
            >
              {t("common.cancel")}
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="text-[12px] text-warnInk mt-2 px-3 py-2 rounded bg-warnSoft border border-warnLine">
          {error}
        </div>
      )}

      {accounts.length > 0 && (
        <>
          <div className={GRP_H + " !mt-4"}>{t("gmail.accounts")}</div>
          <div className={GRP} data-testid="gmail-accounts">
            {accounts.map((a) => (
              <AccountRow key={a.email} a={a} onChanged={onChanged} />
            ))}
          </div>
        </>
      )}

      {accounts.length > 0 && <FiltersGroup c={c} onChanged={onChanged} />}

      <ToolsDisclosure c={c} onChanged={onChanged} />
      <div className={FOOT + " mt-2"}>
        {t("gmail.filters_foot")}
      </div>
    </div>
  );
}

function AccountRow({ a, onChanged }: { a: GmailAccount; onChanged: () => void }) {
  const { t } = useTranslation();
  const [busy, setBusy] = useState(false);
  return (
    <div className={ROW} data-testid={`gmail-account-${a.email}`}>
      <span className="min-w-0 flex-1 flex items-center gap-2">
        <span className="text-[13px] font-medium truncate">{a.email}</span>
        {a.default && <span className={TAG_ACCENT}>{t("connector.default")}</span>}
        {a.needs_reauth && <span className={TAG_WARN}>{t("gmail.sign_in_again")}</span>}
      </span>
      {!a.default && (
        <button
          className="text-[12px] text-muted hover:text-ink shrink-0"
          data-testid={`gmail-make-default-${a.email}`}
          onClick={async () => {
            await setGmailDefaultAccount(a.email);
            onChanged();
          }}
        >
          {t("connector.make_default")}
        </button>
      )}
      <button
        className={XBTN}
        title={t("gmail.disconnect_mailbox_title")}
        data-testid={`gmail-disconnect-${a.email}`}
        disabled={busy}
        onClick={async () => {
          setBusy(true);
          await disconnectGmailAccount(a.email);
          setBusy(false);
          onChanged();
        }}
      >
        ×
      </button>
    </div>
  );
}

function FiltersGroup({ c, onChanged }: Pick<DetailProps, "c" | "onChanged">) {
  const { t } = useTranslation();
  const filters = c.filters ?? { senders: [], labels: [] };
  return (
    <>
      <div className={GRP_H}>{t("gmail.never_show_agents")}</div>
      <div className={GRP} data-testid="gmail-filters">
        <ChipListRow
          label={t("gmail.senders")}
          testid="gmail-filter-senders"
          placeholder={t("gmail.senders_placeholder")}
          values={filters.senders}
          onSave={async (senders) => {
            await setGmailFilters({ senders });
            onChanged();
          }}
        />
        <ChipListRow
          label={t("gmail.labels")}
          testid="gmail-filter-labels"
          placeholder={t("gmail.labels_placeholder")}
          values={filters.labels}
          onSave={async (labels) => {
            await setGmailFilters({ labels });
            onChanged();
          }}
        />
      </div>
      <div className={FOOT}>
        {t("gmail.filters_foot_inner")}
      </div>
    </>
  );
}

function ChipListRow({
  label,
  testid,
  placeholder,
  values,
  onSave,
}: {
  label: string;
  testid: string;
  placeholder: string;
  values: string[];
  onSave: (next: string[]) => Promise<void>;
}) {
  const { t } = useTranslation();
  const [draft, setDraft] = useState("");
  const add = async () => {
    const v = draft.trim();
    if (!v) return;
    setDraft("");
    await onSave([...values, v]);
  };
  return (
    <div className={ROW} data-testid={testid}>
      <span className={LABEL}>{label}</span>
      <span className="min-w-0 flex-1 flex flex-wrap items-center gap-1.5">
        {values.map((v) => (
          <span
            key={v}
            className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-paper border border-line text-[13px]"
          >
            {v}
            <button
              className={XBTN}
              title={t("common.remove")}
              onClick={() => onSave(values.filter((x) => x !== v))}
            >
              ×
            </button>
          </span>
        ))}
        <input
          className="flex-1 min-w-[140px] bg-transparent text-[13px] outline-none placeholder:text-faint"
          placeholder={placeholder}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") add();
          }}
          onBlur={() => draft.trim() && add()}
        />
      </span>
    </div>
  );
}
