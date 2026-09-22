-- Additive: slice work queue used by zseeker2.
-- Safe to run on a live database; does not drop data.

CREATE TABLE IF NOT EXISTS `roots_checked_slice` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `cubic_root_id` int unsigned NOT NULL,
  `zroot_slot` tinyint unsigned NOT NULL COMMENT '1, 2, or 3 — matches zroot1..zroot3',
  `quint_lo` int NOT NULL,
  `quint_hi` int NOT NULL,
  `quart_lo` int NOT NULL,
  `quart_hi` int NOT NULL,
  `worker_id` tinyint unsigned DEFAULT NULL,
  `is_started` tinyint(1) NOT NULL DEFAULT '0',
  `is_finished` tinyint(1) NOT NULL DEFAULT '0',
  `double_hit_count` bigint DEFAULT NULL,
  `float128_hit_count` bigint DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_slice_tile` (`cubic_root_id`,`zroot_slot`,`quint_lo`,`quint_hi`,`quart_lo`,`quart_hi`),
  KEY `idx_slice_pending` (`is_started`,`is_finished`,`id`),
  KEY `idx_slice_cubic` (`cubic_root_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
