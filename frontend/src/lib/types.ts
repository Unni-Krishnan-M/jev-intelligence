export type SignalName = "content" | "collaborative" | "latent" | "popularity" | "preference" | "recency";

export interface SignalValue {
  raw: number;
  normalized: number;
  weight: number;
  contribution: number;
}

export interface User {
  id: number;
  email: string;
  display_name: string;
  is_admin: boolean;
  onboarding_completed: boolean;
  favorite_genres: string[];
  created_at: string;
}

export interface MovieBrief {
  id: number;
  title: string;
  year: number | null;
  genres: string[];
  directors: string[];
  n_ratings: number;
  mean_rating: number | null;
}

export interface MovieDetail extends MovieBrief {
  description: string | null;
  runtime_min: number | null;
  cast: string[];
  keywords: string[];
  tags: string[];
  countries: string[];
  imdb_id: string | null;
  tmdb_id: number | null;
  user_rating: number | null;
  is_favorite: boolean;
  watched: boolean;
  in_model: boolean;
}

export interface MoviePage {
  items: MovieBrief[];
  total: number;
  page: number;
  page_size: number;
}

export interface RecItem {
  recommendation_id: number | null;
  movie_id: number;
  title: string;
  year: number | null;
  genres: string[];
  directors: string[];
  score: number;
  rank: number;
  reason: string;
  reason_code: string;
  secondary_reasons: string[];
  anchor_movie_ids: number[];
  signals: Record<SignalName, SignalValue>;
}

export interface RecResponse {
  items: RecItem[];
  model_version: string;
  generated_at: string;
  request_id: string;
  limit: number;
  offset: number;
  effective_weights: Record<SignalName, number>;
  profile: { interactions: number; genres: number; excluded: number };
  cached: boolean;
}

export interface SimpleItem {
  movie_id: number;
  title: string;
  year: number | null;
  genres: string[];
  directors: string[];
  score: number;
  reason: string | null;
  signals: Record<string, number>;
}

export interface SimpleResponse {
  items: SimpleItem[];
  model_version: string;
  generated_at: string;
  anchor?: MovieBrief | null;
  anchor_kind?: "watched" | "rated" | null;
}

export interface Genre {
  name: string;
  movie_count: number;
}

export interface TasteProfile {
  user: User;
  preferences: { diversity?: "focused" | "balanced" | "adventurous" };
  explicit_genres: string[];
  genre_affinity: { genre: string; count: number }[];
  favorite_directors: { name: string; count: number }[];
  favorite_actors: { name: string; count: number }[];
  rating_histogram: { rating: number; count: number }[];
  mean_rating: number | null;
  counts: { ratings: number; favorites: number; watches: number; negative_feedback: number };
  liked_count: number;
  effective_weights?: Record<SignalName, number>;
  preference_vector?: { genre: string; weight: number }[];
  stage?: "cold" | "warming" | "warm";
  behavioral_ramp?: number;
  model_version?: string;
}

export interface ModelVersion {
  id: number;
  version: string;
  model_type: string;
  dataset_version: string;
  training_seed: number;
  metrics: Record<string, number>;
  artifact_path: string;
  artifact_bytes: number;
  trained_at: string;
  is_active: boolean;
}

export interface ModelVersionDetail extends ModelVersion {
  training_config: Record<string, unknown>;
  manifest: {
    components?: Record<string, Record<string, unknown>>;
    hybrid_config?: Record<string, unknown>;
    artifact_files?: string[];
    als_loss_history?: number[];
    split?: Record<string, unknown>;
    git_commit?: string | null;
    python?: string;
    trained_on_rows?: number;
    n_items?: number;
    error?: string;
  };
}

export interface Metric {
  model_name: string;
  protocol: "test" | "cold_start";
  metric: string;
  k: number;
  value: number;
}

export interface Experiment {
  id: number;
  run_id: string;
  experiment_name: string;
  dataset_version: string;
  model_type: string;
  training_seed: number;
  created_at: string;
  model_version: string | null;
  headline: Record<string, number>;
}

export interface ExperimentDetail extends Experiment {
  parameters: Record<string, unknown>;
  split: Record<string, unknown>;
  summary: {
    seconds_total?: number;
    quick?: boolean;
    n_eval_users?: { test?: number; cold_start?: number };
    selection_metric?: string;
    tuning_trials?: Record<string, number>;
    als_loss_history?: number[];
    evaluation?: { ks?: number[]; cold_start_profile_size?: number; relevance_threshold?: number };
  };
  metrics: Metric[];
}

export interface ActiveSummary {
  model_version: string;
  dataset_version: string;
  n_items: number;
  trained_on_rows: number;
  trained_at: string;
  split: string;
  eval_users: number | null;
  comparison: Record<string, Record<string, number>>;
}
