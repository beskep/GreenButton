/**========================================================================
 MSSQL 지정한 날짜 이후 데이터를 새 DB에 복사.
 한국전력공사 파주지사 DB 복사를 위해 작성.
 *========================================================================**/

/*================================ 변수 입력 ==============================*/

USE [ksem.pajoo.log.filtered]; -- destination

DECLARE @src NVARCHAR(MAX) = 'ksem.pajoo.log' -- source
DECLARE @date NVARCHAR(10) = '2024-07-01'; -- 날짜

-- 복사할 테이블 목록
DECLARE @tables TABLE (TableName NVARCHAR(MAX));
INSERT INTO @tables (TableName)
VALUES ('T_BECO_POINT_CONTROL'),
    ('T_BELO_ALARM_EVENT'),
    ('T_BELO_DC_STATUS'),
    ('T_BELO_ELEC_15MIN'),
    ('T_BELO_ELEC_DAY'),
    ('T_BELO_ELEC_DAY_BAK'),
    ('T_BELO_ELEC_HOUR'),
    ('T_BELO_ELEC_PREDICT'),
    ('T_BELO_ELEC_PREDICT_TEMP'),
    ('T_BELO_ENERGY_15MIN'),
    ('T_BELO_ENERGY_DAY'),
    ('T_BELO_ENERGY_HOUR'),
    ('T_BELO_ENERGY_MONTH'),
    ('T_BELO_ESS_CONTROL'),
    ('T_BELO_FACILITY_15MIN'),
    ('T_BELO_FACILITY_DAY'),
    ('T_BELO_FACILITY_HOUR'),
    ('T_BELO_FACILITY_MONTH'),
    ('T_BELO_RAW_DATA');

DECLARE @table NVARCHAR(MAX);
DECLARE @sql NVARCHAR(MAX);

DECLARE table_cursor CURSOR FOR
SELECT TableName FROM @tables;

/*================================ 실행 ==============================*/

OPEN table_cursor;

FETCH NEXT FROM table_cursor INTO @table;

WHILE @@FETCH_STATUS = 0
BEGIN
    -- 동적 SQL 생성 및 실행
    SET @sql = 'SELECT * INTO ' + QUOTENAME(@table)
        + ' FROM ' + QUOTENAME(@src) + '.dbo.' + @table
        + ' WHERE updateDate >= ' + @date;

    PRINT @sql; -- 디버깅용 SQL 출력
    EXEC sp_executesql @sql; -- 동적 SQL 실행

    FETCH NEXT FROM table_cursor INTO @table;
END;

CLOSE table_cursor;
DEALLOCATE table_cursor;
