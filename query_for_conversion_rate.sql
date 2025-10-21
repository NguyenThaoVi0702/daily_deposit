SELECT 
conversion_date,
conversion_rate
FROM (
SELECT
    *,
    ROW_NUMBER() OVER(
            PARTITION BY conversion_date
            ORDER BY cob_dt DESC 
        ) as rn
FROM (
    SELECT
    DISTINCT
        conversion_date,
        CAST(conversion_rate AS double) AS conversion_rate,
        cob_dt
    FROM
        ogl.ogl_gl_daily_rates
    WHERE
        conversion_date >= '2024-12-09'
        AND conversion_type = '1009'
        AND from_currency = 'VND'
        AND to_currency = 'USD'
)
)
WHERE 
    rn = 1
    
