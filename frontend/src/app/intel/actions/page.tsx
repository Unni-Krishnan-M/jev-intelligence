"use client";

import useSWR from "swr";

import { ActionsBoard } from "@/components/jev/intel/actions-board";
import { useIntelDomain } from "@/components/jev/intel/domain-context";
import { PageHeader } from "@/components/jev/intel/page-header";
import { IntelError, RowsSkeleton } from "@/components/jev/intel/states";
import { EmptyState } from "@/components/jev/states";
import { qs } from "@/lib/api";
import type { Action, Page } from "@/lib/intel-types";

export default function ActionsPage() {
  const { q: dq } = useIntelDomain();
  const { data, error, mutate } = useSWR<Page<Action>>(dq(`/intel/actions${qs({ limit: 100 })}`));
  const items = data?.items ?? [];

  return (
    <div>
      <PageHeader
        eyebrow="decide"
        title="Action plan"
        description="What to do about the decisions, warnings and risks in the latest run, ranked by priority. Each item says why, what it should change and the first step."
        asOf={data?.as_of}
        runId={data?.run_id}
      />
      {error ? (
        <IntelError error={error} retry={() => mutate()} />
      ) : !data ? (
        <RowsSkeleton rows={5} />
      ) : items.length === 0 ? (
        <EmptyState title="No actions" body="Nothing in the latest run calls for an action." />
      ) : (
        <ActionsBoard items={items} />
      )}
    </div>
  );
}
