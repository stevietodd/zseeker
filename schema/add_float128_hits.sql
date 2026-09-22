-- Additive: store float128-refined hit coefficients (LUT indices).
-- Safe to run on a live database; does not drop data.

CREATE TABLE IF NOT EXISTS `float128_hits` (
  `cubic_root_id` int unsigned NOT NULL,
  `zroot_slot` tinyint unsigned NOT NULL COMMENT '1, 2, or 3 — matches zroot1..zroot3',
  `lut_index_5` int NOT NULL,
  `lut_index_4` int NOT NULL,
  `lut_index_3` int NOT NULL,
  `lut_index_2` int NOT NULL,
  `lut_index_1` int NOT NULL,
  `lut_index_0` int NOT NULL,
  KEY `idx_float128_hits_cubic` (`cubic_root_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
