loan daily:

WITH
sum_by_cob_and_crcd AS (
SELECT
	cob_dt,
	crcd,
	open_bal AS org_open_bal,
	CASE 
		WHEN crcd = 'VND'
		THEN open_bal
		ELSE open_bal * conversion_rate
	END AS vnd_open_bal
FROM
	(
	SELECT
		cob_dt,
		crcd,
		sum(open_bal) AS open_bal
	FROM
		datalake.test.ln_summary
	WHERE
		LENGTH(TYPE) = 4
		AND CAST(TYPE AS int) NOT IN (9855, 9725, 8906, 8907, 8908, 8909, 8910, 8911, 7000, 7001, 7002, 7003, 7004, 7005, 7007, 7008)
	GROUP BY
		cob_dt,
		crcd
	) loan
LEFT JOIN (
	SELECT 
	conversion_date,
    from_currency,
    conversion_rate
FROM (
SELECT
    *,
    ROW_NUMBER() OVER(
            PARTITION BY conversion_date, from_currency 
            ORDER BY cob_dt DESC 
        ) as rn
FROM (
    SELECT
    DISTINCT
        conversion_date,
        from_currency,
        CAST(conversion_rate AS double) AS conversion_rate,
        cob_dt
    FROM
        ogl.ogl_gl_daily_rates
    WHERE
        conversion_date >= '2024-12-09'
        AND conversion_type = '1009'
        AND to_currency = 'VND'
)
)
WHERE 
    rn = 1
	) conv
	ON
	loan.crcd = conv.from_currency
	AND loan.cob_dt = conv.conversion_date
)
SELECT
	*
FROM
	(
(
	SELECT
		cob_dt,
		crcd AS TYPE,
		org_open_bal,
		vnd_open_bal
	FROM
		sum_by_cob_and_crcd
	WHERE
		crcd IN ('VND', 'USD')
)
UNION (
SELECT
	cob_dt,
	'SUM' AS TYPE,
	0 AS org_open_bal,
	sum(vnd_open_bal) AS vnd_open_bal
FROM
	sum_by_cob_and_crcd
GROUP BY
	cob_dt
)
)
ORDER BY
	cob_dt,
	TYPE ASC


=======================================

depst daily:

WITH
sum_by_cob_and_crcd AS (
SELECT
	cob_dt,
	crcd,
	open_bal AS org_open_bal,
	CASE 
		WHEN crcd = 'VND'
		THEN open_bal
		ELSE open_bal * conversion_rate
	END AS vnd_open_bal
FROM
	(
	SELECT
		cob_dt,
		crcd,
		sum(open_bal) AS open_bal
	FROM
		datalake.test.depst_summary
	WHERE
		LENGTH(TYPE) = 4
		AND (
			(
				grp IN ('DDA', 'SAV', 'CD')
			AND CAST(TYPE AS int) NOT IN (1105, 1200, 1204, 1205, 1206, 1207, 1219, 1300, 1302, 2106, 2109, 2702, 2703, 2707, 2708, 2709, 2710, 2712, 2713, 2715, 2924)
			)
			OR (grp = 'WASH'
				AND TYPE = '4409')
		)
	GROUP BY
		cob_dt,
		crcd
	) loan
LEFT JOIN (
	SELECT 
	conversion_date,
    from_currency,
    conversion_rate
	FROM (
	SELECT
	    *,
	    ROW_NUMBER() OVER(
	            PARTITION BY conversion_date, from_currency 
	            ORDER BY cob_dt DESC 
	        ) as rn
	FROM (
	    SELECT
	    DISTINCT
	        conversion_date,
	        from_currency,
	        CAST(conversion_rate AS double) AS conversion_rate,
	        cob_dt
	    FROM
	        ogl.ogl_gl_daily_rates
	    WHERE
	        conversion_date >= '2024-12-09'
	        AND conversion_type = '1009'
	        AND to_currency = 'VND'
	)
	)
	WHERE 
	    rn = 1
	) conv
	ON
	loan.crcd = conv.from_currency
	AND loan.cob_dt = conv.conversion_date
)
SELECT
	*
FROM
	(
(
	SELECT
		cob_dt,
		crcd AS TYPE,
		org_open_bal,
		vnd_open_bal
	FROM
		sum_by_cob_and_crcd
	WHERE
		crcd IN ('VND', 'USD')
)
UNION (
SELECT
	cob_dt,
	'SUM' AS TYPE,
	NULL AS org_open_bal,
	sum(vnd_open_bal) AS vnd_open_bal
FROM
	sum_by_cob_and_crcd
GROUP BY
	cob_dt
)
)
ORDER BY
	cob_dt,
	TYPE
