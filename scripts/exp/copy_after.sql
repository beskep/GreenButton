/**========================================================================
  MSSQL 지정한 날짜 이후 데이터를 새 DB에 복사.
  *========================================================================**/

-- 데이터베이스 이름 변수 설정 (sysname으로 유니코드 지원)
DECLARE @sourceDBName sysname = N'ksem.pajoo.log';
DECLARE @targetDBName sysname = N'filtered'; -- 수정

-- 복사할 특정 날짜 설정
DECLARE @cutoffDate DATETIME = '2024-06-01'; -- 수정

-- 복사할 테이블 목록 (쉼표로 구분)
DECLARE @tableList VARCHAR(MAX) = 'T_BELO_ELEC_15MIN,T_BELO_ELEC_DAY,T_BELO_ELEC_HOUR,'
     + 'T_BELO_ENERGY_15MIN,T_BELO_ENERGY_DAY,T_BELO_ENERGY_HOUR,T_BELO_ENERGY_MONTH,'
     + 'T_BELO_FACILITY_15MIN,T_BELO_FACILITY_DAY,T_BELO_FACILITY_HOUR,T_BELO_FACILITY_MONTH';

/*================================ 실행 ==============================*/

DECLARE @tableName sysname;
DECLARE @sql NVARCHAR(MAX);
DECLARE @params NVARCHAR(100) = N'@cutoffDate DATETIME';
DECLARE @quotedSourceDBName sysname = QUOTENAME(@sourceDBName);
DECLARE @quotedTargetDBName sysname = QUOTENAME(@targetDBName);

-- STRING_SPLIT 결과를 사용하여 각 테이블에 대해 반복 작업 수행
DECLARE table_cursor CURSOR LOCAL FAST_FORWARD FOR
SELECT value FROM STRING_SPLIT(@tableList, ',') WHERE value != '';

OPEN table_cursor;
FETCH NEXT FROM table_cursor INTO @tableName;

WHILE @@FETCH_STATUS = 0
BEGIN
    BEGIN TRY
        -- 동적 SQL을 사용하여 테이블 존재 여부 확인 및 생성
        SET @sql = N'
        -- 대상 DB에 테이블이 존재하는지 확인 (INFORMATION_SCHEMA 사용)
        IF NOT EXISTS (SELECT 1 FROM ' + @quotedTargetDBName + '.INFORMATION_SCHEMA.TABLES 
                       WHERE TABLE_SCHEMA = ''dbo'' AND TABLE_NAME = @tableName)
        BEGIN
            PRINT N''대상 DB에 테이블 '' + @tableName + ''이(가) 존재하지 않습니다. 새로 생성합니다.'';
            -- 원본 테이블의 스키마를 복사하여 새로운 테이블 생성
            SELECT * INTO ' + @quotedTargetDBName + '.dbo.' + QUOTENAME(@tableName) + '
            FROM ' + @quotedSourceDBName + '.dbo.' + QUOTENAME(@tableName) + '
            WHERE 1 = 0; -- WHERE 1=0 조건을 사용하여 스키마만 복사
        END;

        PRINT N''테이블 '' + @tableName + '' 복사 시작...'';
        -- 지정된 날짜 이후의 데이터만 대상 DB의 테이블로 복사
        INSERT INTO ' + @quotedTargetDBName + '.dbo.' + QUOTENAME(@tableName) + '
        SELECT * FROM ' + @quotedSourceDBName + '.dbo.' + QUOTENAME(@tableName) + '
        WHERE updateDate >= @cutoffDate;

        PRINT N''테이블 '' + @tableName + '' 복사 완료'';
        ';

        SET @params = N'@cutoffDate DATETIME, @tableName sysname';

        EXEC sp_executesql @sql, @params,
            @cutoffDate = @cutoffDate,
            @tableName = @tableName;

    END TRY
    BEGIN CATCH
        -- 오류 로그 출력
        PRINT N'테이블 ' + @tableName + ' 복사 실패: ' + ERROR_MESSAGE();
    END CATCH;

    FETCH NEXT FROM table_cursor INTO @tableName;
END;

CLOSE table_cursor;
DEALLOCATE table_cursor;

PRINT N'스크립트 실행 완료';