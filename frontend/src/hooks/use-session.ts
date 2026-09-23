"use client";

import useSWR, { useSWRConfig } from "swr";

import { api, ApiError } from "@/lib/api";
import type { User } from "@/lib/types";

async function fetchMe(): Promise<User | null> {
  try {
    return await api<User>("/users/me");
  } catch (e) {
    if (e instanceof ApiError && e.status === 401) return null;
    throw e;
  }
}

export function useSession() {
  const { data, error, isLoading, mutate } = useSWR<User | null>("session:me", fetchMe, { revalidateOnFocus: true });
  const { mutate: globalMutate } = useSWRConfig();

  async function logout() {
    await api("/auth/logout", { method: "POST" });
    await mutate(null, { revalidate: false });
    // drop every cached personal response
    await globalMutate(() => true, undefined, { revalidate: false });
  }

  return { user: data ?? null, loading: isLoading, error, refresh: mutate, logout };
}
