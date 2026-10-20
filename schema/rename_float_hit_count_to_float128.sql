-- Rename leftover float hit-count columns to float128.
--
-- MySQL 8 RENAME COLUMN is metadata-only (does not copy roots_checked).
-- Existing values are old float tallies, not float128 counts.
-- Column order is left unchanged.

ALTER TABLE roots_checked
  RENAME COLUMN float_hit_count1 TO float128_hit_count1,
  RENAME COLUMN float_hit_count2 TO float128_hit_count2,
  RENAME COLUMN float_hit_count3 TO float128_hit_count3,
  ALGORITHM=INSTANT;
