-- Rename leftover float hit-count columns to float128 and place them after
-- the matching double_hit_count columns.
--
-- CHANGE ... AFTER may copy the table. roots_checked is large; run this
-- during a maintenance window. Existing values are old float tallies and
-- are not float128 counts — leave them or NULL them as you prefer.

ALTER TABLE roots_checked
  CHANGE COLUMN float_hit_count1 float128_hit_count1 INT DEFAULT NULL AFTER double_hit_count1,
  CHANGE COLUMN float_hit_count2 float128_hit_count2 INT DEFAULT NULL AFTER double_hit_count2,
  CHANGE COLUMN float_hit_count3 float128_hit_count3 INT DEFAULT NULL AFTER double_hit_count3;

ALTER TABLE roots_checked_slice
  CHANGE COLUMN float_hit_count float128_hit_count BIGINT DEFAULT NULL AFTER double_hit_count;
