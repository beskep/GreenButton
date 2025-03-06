/**========================================================================
 MSSQL 지정한 날짜 이후 데이터를 새 DB에 복사.
 한국전력공사 파주지사 DB 복사를 위해 작성.
 *========================================================================**/

-- 데이터베이스 이름 변수 설정
DECLARE @sourceDB VARCHAR(255) = 'ksem.pajoo.log';
DECLARE @targetDB VARCHAR(255) = 'ksem.pajoo.log20240601'; -- 수정

-- 복사할 특정 날짜 설정
DECLARE @cutoffDate DATETIME = '2024-06-01'; -- 수정

-- 복사할 테이블 목록 (쉼표로 구분)
DECLARE @tableList VARCHAR(MAX) = 'T_BELO_ELEC_15MIN,T_BELO_ELEC_DAY,T_BELO_ELEC_HOUR,'
    + 'T_BELO_ENERGY_15MIN,T_BELO_ENERGY_DAY,T_BELO_ENERGY_HOUR,T_BELO_ENERGY_MONTH'
    + 'T_BELO_FACILITY_15MIN,T_BELO_FACILITY_DAY,T_BELO_FACILITY_HOUR,T_BELO_FACILITY_MONTH';

/*================================ 실행 ==============================*/

-- 테이블 목록 분할
DECLARE @tableName VARCHAR(255);
DECLARE @tableIndex INT = 1;
DECLARE @tableCount INT;

-- 테이블 개수 계산
SELECT @tableCount = COUNT(*)
FROM STRING_SPLIT(@tableList, ',');

-- 테이블 목록 순회
WHILE @tableIndex <= @tableCount
BEGIN
   -- 테이블 이름 가져오기
   SELECT @tableName = value
   FROM (SELECT value, ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS RowNum FROM STRING_SPLIT(@tableList, ',')) AS Subquery
   WHERE RowNum = @tableIndex;

   -- 테이블 복사 및 로그 출력
   BEGIN TRY
        -- 테이블이 존재하는지 확인
        IF OBJECT_ID(QUOTENAME(@targetDB) + '.dbo.' + QUOTENAME(@tableName), 'U') IS NOT NULL
        BEGIN
            -- 동적 SQL 생성
            DECLARE @sql NVARCHAR(MAX);
            SET @sql = 'INSERT INTO ' + QUOTENAME(@targetDB) + '.dbo.' + QUOTENAME(@tableName)
                + ' SELECT * FROM ' + QUOTENAME(@sourceDB) + '.dbo.' + QUOTENAME(@tableName)
                + ' WHERE updateDate >= ''' + CONVERT(VARCHAR, @cutoffDate, 120) + '''';

            -- 동적 SQL 실행
            EXEC (@sql);

            -- 로그 출력
            PRINT '테이블 ' + @tableName + ' 복사 완료';
        END
        ELSE
        BEGIN
            PRINT '테이블 ' + @tableName + ' 이 존재하지 않습니다.';
        END
   END TRY
   BEGIN CATCH
      -- 오류 로그 출력
      PRINT '테이블 ' + @tableName + ' 복사 실패: ' + ERROR_MESSAGE();
   END CATCH;

   -- 다음 테이블로 이동
   SET @tableIndex = @tableIndex + 1;
END;