
--
-- Destructive DDL: this file DROPs tables (data loss). It will not run those
-- drops unless you opt in for this session first:
--   SET @zseeker_allow_schema_drops := 1;
-- Then SOURCE this file (or pipe the same SET before the script body).
-- If you omit that, execution stops immediately with an error explaining why.
--

DELIMITER ;;
DROP PROCEDURE IF EXISTS `zseeker_schema_guard` ;;
CREATE PROCEDURE `zseeker_schema_guard`()
BEGIN
  IF IFNULL(@zseeker_allow_schema_drops, 0) <> 1 THEN
    SIGNAL SQLSTATE '45000'
      SET MESSAGE_TEXT = 'schema.sql: DROP TABLE blocked. Read the header comment, then run SET @zseeker_allow_schema_drops := 1; in this session and re-apply.';
  END IF;
END ;;
DELIMITER ;
CALL `zseeker_schema_guard`();
DROP PROCEDURE IF EXISTS `zseeker_schema_guard`;
