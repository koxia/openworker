import { useMemo, useState } from "react";
import { Icon } from "./Icon";

export interface Option {
  value: string;
  label: string;
  description?: string;
}

interface Props {
  prefix?: string;
  value: string;
  options: Option[];
  onChange: (value: string) => void;
  align?: "left" | "right";
  // Extra classes appended to the trigger pill (e.g. "chip" for a bordered composer-head chip).
  className?: string;
  searchable?: boolean;
  searchPlaceholder?: string;
}

export function Dropdown({
  prefix,
  value,
  options,
  onChange,
  align = "left",
  className,
  searchable = false,
  searchPlaceholder = "Search models…",
}: Props) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => {
    if (!searchable || !query.trim()) return options;
    const pattern = query.trim().toLowerCase();
    const glob = pattern.includes("*") || pattern.includes("?") ? pattern : `*${pattern}*`;
    // Escape regex syntax but leave glob wildcards for their own translation.
    const escaped = glob.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, ".*").replace(/\?/g, ".");
    const matcher = new RegExp(`^${escaped}$`);
    return options.filter((option) => matcher.test(option.label.toLowerCase()) || matcher.test(option.value.toLowerCase()));
  }, [options, query, searchable]);
  const current = options.find((o) => o.value === value);
  const label = (prefix ? `${prefix}: ` : "") + (current?.label || value);
  return (
    <div className="dd">
      <button
        className={"pill" + (className ? " " + className : "")}
        onClick={() => setOpen((v) => !v)}
        title={label}
      >
        <span className="pill-label">{label}</span>
        <Icon name="chevronDown" size={13} className="caret" />
      </button>
      {open && (
        <>
          <div className="dd-backdrop" onClick={() => setOpen(false)} />
          <div className={"dd-menu " + align}>
            {filtered.map((o) => (
              <div
                key={o.value}
                className={"dd-item" + (o.value === value ? " sel" : "")}
                onClick={() => {
                  onChange(o.value);
                  setOpen(false);
                }}
              >
                <div className="dd-label">
                  {o.label}
                  {o.value === value && <span className="chk">✓</span>}
                </div>
                {o.description && <div className="dd-desc">{o.description}</div>}
              </div>
            ))}
            {filtered.length === 0 && <div className="dd-empty">No matching models</div>}
            {searchable && (
              <input
                autoFocus
                className="dd-search"
                placeholder={searchPlaceholder}
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onClick={(event) => event.stopPropagation()}
              />
            )}
          </div>
        </>
      )}
    </div>
  );
}
