"use client";

import { ArrowLeft, Plus, Trash2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { useId, useState } from "react";
import { toast } from "sonner";
import useSWR from "swr";

import { Panel } from "@/components/jev/admin/ui";
import Link from "@/components/jev/intel/domain-context";
import { PageHeader } from "@/components/jev/intel/page-header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { api, errorMessage } from "@/lib/api";
import { experimentFormProblems, METRIC_LABEL, weightShares } from "@/lib/ops";
import type { ExperimentCreate, ExperimentOut, GovernedModelList, GuardrailIn, GuardrailMetric, PrimaryMetric, VariantIn } from "@/lib/ops-types";

const PRIMARY: PrimaryMetric[] = ["interaction_rate", "positive_rate", "rating_rate", "feedback_rate", "ndcg_at_10", "diversity", "novelty"];
const GUARD_METRICS: GuardrailMetric[] = ["negative_rate", "latency_p95_ms", ...PRIMARY, "coverage"];
type Bound = "max_increase" | "max_decrease" | "max_ratio";
const BOUND_LABEL: Record<Bound, string> = { max_increase: "max increase", max_decrease: "max decrease", max_ratio: "max ratio ×" };
const CHAMPION = "__champion__";

interface VariantForm {
  name: string;
  is_control: boolean;
  weight: string;
  description: string;
  lambda: string;
  halfLife: string;
  strategy: boolean;
  model: string;
}
interface GuardForm {
  metric: GuardrailMetric;
  bound: Bound;
  value: string;
}

const variant = (name: string, control = false): VariantForm => ({ name, is_control: control, weight: "1", description: "", lambda: "", halfLife: "", strategy: true, model: CHAMPION });

function Field({ label, children, hint, htmlFor }: { label: string; children: React.ReactNode; hint?: string; htmlFor: string }) {
  return (
    <div className="min-w-0 space-y-1">
      <Label htmlFor={htmlFor} className="text-xs font-normal text-muted-foreground">{label}</Label>
      {children}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

const num = (s: string) => (s.trim() === "" ? null : Number(s));

export default function NewExperimentPage() {
  const id = useId();
  const router = useRouter();
  const models = useSWR<GovernedModelList>("/governance/models");
  const [key, setKey] = useState("");
  const [name, setName] = useState("");
  const [hypothesis, setHypothesis] = useState("");
  const [primary, setPrimary] = useState<PrimaryMetric>("interaction_rate");
  const [traffic, setTraffic] = useState("100");
  const [windowH, setWindowH] = useState("");
  const [alpha, setAlpha] = useState("0.05");
  const [power, setPower] = useState("0.8");
  const [mde, setMde] = useState("0.1");
  const [minUsers, setMinUsers] = useState("100");
  const [variants, setVariants] = useState<VariantForm[]>([variant("control", true), variant("treatment")]);
  const [guards, setGuards] = useState<GuardForm[]>([
    { metric: "negative_rate", bound: "max_increase", value: "0.02" },
    { metric: "latency_p95_ms", bound: "max_ratio", value: "1.5" },
  ]);
  const [busy, setBusy] = useState(false);
  const [tried, setTried] = useState(false);

  const setV = (i: number, patch: Partial<VariantForm>) => setVariants((vs) => vs.map((v, j) => (j === i ? { ...v, ...patch } : v)));
  const setControl = (i: number) => setVariants((vs) => vs.map((v, j) => ({ ...v, is_control: j === i })));
  const setG = (i: number, patch: Partial<GuardForm>) => setGuards((gs) => gs.map((g, j) => (j === i ? { ...g, ...patch } : g)));
  const shares = weightShares(variants.map((v) => Number(v.weight) || 0));
  const problems = experimentFormProblems({ key, name, variants: variants.map((v) => ({ name: v.name, is_control: v.is_control, weight: Number(v.weight) })) });
  const versions = (models.data?.items ?? []).filter((m) => m.state !== "rejected").map((m) => m.version);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setTried(true);
    if (problems.length) return;
    const body: ExperimentCreate = {
      key,
      name: name.trim(),
      hypothesis: hypothesis.trim(),
      primary_metric: primary,
      traffic_percent: Number(traffic),
      attribution_window_hours: num(windowH),
      analysis: { alpha: Number(alpha), power: Number(power), mde_relative: Number(mde), min_users_per_variant: Number(minUsers) },
      guardrails: guards.map((g): GuardrailIn => ({ metric: g.metric, [g.bound]: Number(g.value) })),
      variants: variants.map((v): VariantIn => {
        const overrides: Record<string, unknown> = {};
        if (num(v.lambda) !== null) overrides.diversity_lambda = num(v.lambda);
        return {
          name: v.name,
          is_control: v.is_control,
          weight: Number(v.weight),
          description: v.description.trim(),
          config: {
            hybrid_overrides: overrides,
            model_version: v.model === CHAMPION ? null : v.model,
            strategy_decision: v.strategy,
            recency_half_life_days: num(v.halfLife),
          },
        };
      }),
    };
    setBusy(true);
    try {
      const out = await api<ExperimentOut>("/experiments/online", { json: body });
      toast.success(`Draft ${out.key} created`);
      router.push(`/intel/ops/experiments/${encodeURIComponent(out.key)}`);
    } catch (err) {
      toast.error(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <Link href="/intel/ops/experiments" className="mb-6 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="size-4" aria-hidden /> Experiments
      </Link>
      <PageHeader eyebrow="operations" title="New experiment" description="A draft serves nobody. Review it, then start it from its page. After start the configuration is immutable; only the traffic ramp changes." />
      <form onSubmit={submit} className="space-y-4" noValidate>
        <Panel title="What is being tested">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <Field label="Key (URL id)" htmlFor={`${id}-key`} hint="lower-case, 3–64 chars, e.g. mmr-lambda-06">
              <Input id={`${id}-key`} value={key} onChange={(e) => setKey(e.target.value.toLowerCase())} className="font-mono" required autoComplete="off" />
            </Field>
            <Field label="Name" htmlFor={`${id}-name`}>
              <Input id={`${id}-name`} value={name} onChange={(e) => setName(e.target.value)} required maxLength={200} />
            </Field>
            <div className="md:col-span-2">
              <Field label="Hypothesis" htmlFor={`${id}-hyp`}>
                <textarea id={`${id}-hyp`} value={hypothesis} onChange={(e) => setHypothesis(e.target.value)} rows={2} maxLength={5000} className="w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50" placeholder="A lower MMR λ raises intra-list diversity without lowering interactions." />
              </Field>
            </div>
            <Field label="Primary metric" htmlFor={`${id}-pm`} hint="Tested at a Bonferroni-adjusted α per treatment.">
              <Select value={primary} onValueChange={(v) => setPrimary(v as PrimaryMetric)}>
                <SelectTrigger id={`${id}-pm`} className="w-full"><SelectValue /></SelectTrigger>
                <SelectContent>{PRIMARY.map((m) => <SelectItem key={m} value={m}>{METRIC_LABEL[m]}</SelectItem>)}</SelectContent>
              </Select>
            </Field>
            <div className="grid grid-cols-2 gap-4">
              <Field label="Traffic enrolled (%)" htmlFor={`${id}-tr`}>
                <Input id={`${id}-tr`} type="number" min={0} max={100} step={1} value={traffic} onChange={(e) => setTraffic(e.target.value)} className="num" />
              </Field>
              <Field label="Attribution window (h)" htmlFor={`${id}-win`} hint="empty = API default">
                <Input id={`${id}-win`} type="number" min={1} max={720} value={windowH} onChange={(e) => setWindowH(e.target.value)} className="num" />
              </Field>
            </div>
          </div>
        </Panel>

        <Panel title="Variants" action={<Button type="button" size="sm" variant="outline" disabled={variants.length >= 5} onClick={() => setVariants((vs) => [...vs, variant(`treatment-${vs.length}`)])}><Plus aria-hidden /> Add variant</Button>}>
          <div className="space-y-4" role="radiogroup" aria-label="Control variant">
            {variants.map((v, i) => (
              <fieldset key={i} className="rounded border hairline p-3">
                <legend className="px-1 text-xs text-muted-foreground">Variant {i + 1} · {(shares[i] * 100).toFixed(0)} % of enrolled traffic</legend>
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  <Field label="Name" htmlFor={`${id}-v${i}-name`}>
                    <Input id={`${id}-v${i}-name`} value={v.name} onChange={(e) => setV(i, { name: e.target.value.toLowerCase() })} className="font-mono" />
                  </Field>
                  <Field label="Weight" htmlFor={`${id}-v${i}-w`}>
                    <Input id={`${id}-v${i}-w`} type="number" min={0.01} max={1000} step="any" value={v.weight} onChange={(e) => setV(i, { weight: e.target.value })} className="num" />
                  </Field>
                  <Field label="MMR λ override (0–1)" htmlFor={`${id}-v${i}-l`} hint="empty = champion's">
                    <Input id={`${id}-v${i}-l`} type="number" min={0} max={1} step={0.05} value={v.lambda} onChange={(e) => setV(i, { lambda: e.target.value })} className="num" />
                  </Field>
                  <Field label="Recency half-life (days)" htmlFor={`${id}-v${i}-h`} hint="empty = off">
                    <Input id={`${id}-v${i}-h`} type="number" min={1} max={36500} value={v.halfLife} onChange={(e) => setV(i, { halfLife: e.target.value })} className="num" />
                  </Field>
                  <Field label="Model" htmlFor={`${id}-v${i}-m`}>
                    <Select value={v.model} onValueChange={(m) => setV(i, { model: m })}>
                      <SelectTrigger id={`${id}-v${i}-m`} className="w-full font-mono text-xs"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value={CHAMPION}>active model (champion)</SelectItem>
                        {versions.map((x) => <SelectItem key={x} value={x} className="font-mono text-xs">{x}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </Field>
                  <div className="sm:col-span-2">
                    <Field label="Description" htmlFor={`${id}-v${i}-d`}>
                      <Input id={`${id}-v${i}-d`} value={v.description} onChange={(e) => setV(i, { description: e.target.value })} maxLength={300} />
                    </Field>
                  </div>
                  <div className="flex flex-wrap items-end justify-between gap-3">
                    <label className="flex items-center gap-2 text-sm">
                      <Switch checked={v.strategy} onCheckedChange={(c) => setV(i, { strategy: c })} aria-label={`Strategy decision for ${v.name}`} />
                      strategy {v.strategy ? "on" : "off"}
                    </label>
                    <label className="flex items-center gap-2 text-sm">
                      <input type="radio" name={`${id}-control`} checked={v.is_control} onChange={() => setControl(i)} className="size-4 accent-[var(--primary)]" />
                      control
                    </label>
                    {variants.length > 2 && (
                      <Button type="button" size="icon" variant="ghost" onClick={() => setVariants((vs) => vs.filter((_, j) => j !== i))} aria-label={`Remove variant ${v.name}`}><Trash2 aria-hidden /></Button>
                    )}
                  </div>
                </div>
              </fieldset>
            ))}
          </div>
        </Panel>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Panel title="Guardrails" action={<Button type="button" size="sm" variant="outline" disabled={guards.length >= 10} onClick={() => setGuards((gs) => [...gs, { metric: "positive_rate", bound: "max_decrease", value: "0.02" }])}><Plus aria-hidden /> Add</Button>}>
            <p className="mb-3 text-xs text-muted-foreground">Absolute bounds on treatment − control (rates as fractions), or a ratio for latency. A significant harmful move also breaches.</p>
            <ul className="space-y-2">
              {guards.map((g, i) => (
                <li key={i} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_80px_auto] items-center gap-2">
                  <Select value={g.metric} onValueChange={(m) => setG(i, { metric: m as GuardrailMetric, bound: m === "latency_p95_ms" ? "max_ratio" : g.bound === "max_ratio" ? "max_decrease" : g.bound })}>
                    <SelectTrigger className="w-full text-xs" aria-label={`Guardrail ${i + 1} metric`}><SelectValue /></SelectTrigger>
                    <SelectContent>{GUARD_METRICS.map((m) => <SelectItem key={m} value={m}>{METRIC_LABEL[m] ?? m}</SelectItem>)}</SelectContent>
                  </Select>
                  <Select value={g.bound} onValueChange={(b) => setG(i, { bound: b as Bound })}>
                    <SelectTrigger className="w-full text-xs" aria-label={`Guardrail ${i + 1} bound`}><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {(g.metric === "latency_p95_ms" ? (["max_ratio"] as Bound[]) : (["max_increase", "max_decrease"] as Bound[])).map((b) => <SelectItem key={b} value={b}>{BOUND_LABEL[b]}</SelectItem>)}
                    </SelectContent>
                  </Select>
                  <Input type="number" step="any" min={0} value={g.value} onChange={(e) => setG(i, { value: e.target.value })} className="num h-9" aria-label={`Guardrail ${i + 1} threshold`} />
                  <Button type="button" size="icon" variant="ghost" onClick={() => setGuards((gs) => gs.filter((_, j) => j !== i))} aria-label={`Remove guardrail ${i + 1}`}><Trash2 aria-hidden /></Button>
                </li>
              ))}
            </ul>
          </Panel>
          <Panel title="Analysis">
            <div className="grid grid-cols-2 gap-4">
              <Field label="α (two-sided)" htmlFor={`${id}-a`}><Input id={`${id}-a`} type="number" min={0.001} max={0.2} step={0.01} value={alpha} onChange={(e) => setAlpha(e.target.value)} className="num" /></Field>
              <Field label="Power" htmlFor={`${id}-p`}><Input id={`${id}-p`} type="number" min={0.5} max={0.99} step={0.05} value={power} onChange={(e) => setPower(e.target.value)} className="num" /></Field>
              <Field label="MDE (relative)" htmlFor={`${id}-mde`} hint="0.1 = a 10 % relative change"><Input id={`${id}-mde`} type="number" min={0.01} max={10} step={0.01} value={mde} onChange={(e) => setMde(e.target.value)} className="num" /></Field>
              <Field label="Min members per variant" htmlFor={`${id}-mu`}><Input id={`${id}-mu`} type="number" min={2} step={1} value={minUsers} onChange={(e) => setMinUsers(e.target.value)} className="num" /></Field>
            </div>
            <p className="mt-3 text-xs text-muted-foreground">Below the minimum, or without a significant primary effect, the conclusion is inconclusive and no variant is shipped.</p>
          </Panel>
        </div>

        {tried && problems.length > 0 && (
          <ul role="alert" className="space-y-1 rounded-lg border border-destructive/40 bg-destructive/5 px-4 py-3 text-sm">
            {problems.map((p) => <li key={p}>{p}</li>)}
          </ul>
        )}
        <div className="flex flex-wrap justify-end gap-2">
          <Button type="button" variant="outline" asChild><Link href="/intel/ops/experiments">Cancel</Link></Button>
          <Button type="submit" disabled={busy}>{busy ? "Creating…" : "Create draft"}</Button>
        </div>
      </form>
    </div>
  );
}
