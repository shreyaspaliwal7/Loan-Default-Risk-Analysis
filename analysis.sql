-- query 1: overall default rate across portfolio
SELECT 
    COUNT(*) AS total_loans,
    SUM(default_flag) AS total_defaulters,
    ROUND(AVG(default_flag) * 100.0, 2) AS default_rate_pct
FROM repayment_status;

-- query 2: default rate grouped by loan purpose
SELECT 
    l.loan_purpose,
    COUNT(l.loan_id) AS total_loans,
    SUM(r.default_flag) AS defaulter_count,
    ROUND(AVG(r.default_flag) * 100.0, 2) AS default_rate_pct
FROM loans l
JOIN repayment_status r ON l.loan_id = r.loan_id
GROUP BY l.loan_purpose
ORDER BY default_rate_pct DESC;

-- query 3: default rate by age band
SELECT 
    CASE 
        WHEN b.age < 25 THEN 'under 25'
        WHEN b.age BETWEEN 25 AND 39 THEN '25-39'
        WHEN b.age BETWEEN 40 AND 59 THEN '40-59'
        ELSE '60 and above'
    END AS age_band,
    COUNT(l.loan_id) AS total_loans,
    SUM(r.default_flag) AS defaulter_count,
    ROUND(AVG(r.default_flag) * 100.0, 2) AS default_rate_pct
FROM borrowers b
JOIN loans l ON b.borrower_id = l.borrower_id
JOIN repayment_status r ON l.loan_id = r.loan_id
GROUP BY age_band
ORDER BY default_rate_pct DESC;

-- query 4: default rate by income band
SELECT 
    CASE 
        WHEN b.income < 25000 THEN 'low (<25k)'
        WHEN b.income BETWEEN 25000 AND 50000 THEN 'medium (25k-50k)'
        WHEN b.income BETWEEN 50001 AND 100000 THEN 'high (50k-100k)'
        ELSE 'very high (>100k)'
    END AS income_band,
    COUNT(l.loan_id) AS total_loans,
    SUM(r.default_flag) AS defaulter_count,
    ROUND(AVG(r.default_flag) * 100.0, 2) AS default_rate_pct
FROM borrowers b
JOIN loans l ON b.borrower_id = l.borrower_id
JOIN repayment_status r ON l.loan_id = r.loan_id
GROUP BY income_band
ORDER BY default_rate_pct DESC;

-- query 5: top 5 riskiest segments with minimum 15 loans using having
SELECT 
    l.loan_purpose,
    l.credit_history,
    COUNT(l.loan_id) AS total_loans,
    SUM(r.default_flag) AS defaulter_count,
    ROUND(AVG(r.default_flag) * 100.0, 2) AS default_rate_pct
FROM loans l
JOIN repayment_status r ON l.loan_id = r.loan_id
GROUP BY l.loan_purpose, l.credit_history
HAVING COUNT(l.loan_id) >= 15
ORDER BY default_rate_pct DESC
LIMIT 5;

-- query 6: average loan amount and duration for defaulters vs non-defaulters
SELECT 
    r.default_flag,
    COUNT(l.loan_id) AS total_loans,
    ROUND(AVG(l.loan_amount), 2) AS avg_loan_amount,
    ROUND(AVG(l.loan_duration), 1) AS avg_duration_months,
    ROUND(AVG(b.income), 2) AS avg_borrower_income
FROM loans l
JOIN repayment_status r ON l.loan_id = r.loan_id
JOIN borrowers b ON l.borrower_id = b.borrower_id
GROUP BY r.default_flag;

-- query 7: conditional aggregation to count defaulted vs non-defaulted loans per credit history
SELECT 
    l.credit_history,
    COUNT(l.loan_id) AS total_loans,
    SUM(CASE WHEN r.default_flag = 1 THEN 1 ELSE 0 END) AS defaulters,
    SUM(CASE WHEN r.default_flag = 0 THEN 1 ELSE 0 END) AS non_defaulters,
    ROUND(AVG(r.default_flag) * 100.0, 2) AS default_rate_pct
FROM loans l
JOIN repayment_status r ON l.loan_id = r.loan_id
GROUP BY l.credit_history
ORDER BY default_rate_pct DESC;

-- query 8: subquery finding borrowers exceeding average loan amount of their loan purpose
SELECT 
    l.loan_id,
    l.borrower_id,
    l.loan_purpose,
    l.loan_amount,
    p.avg_purpose_amount
FROM loans l
JOIN (
    SELECT loan_purpose, ROUND(AVG(loan_amount), 2) AS avg_purpose_amount
    FROM loans
    GROUP BY loan_purpose
) p ON l.loan_purpose = p.loan_purpose
WHERE l.loan_amount > p.avg_purpose_amount
ORDER BY l.loan_purpose, l.loan_amount DESC
LIMIT 15;

-- query 9: cte ranking loan purposes by default rate
WITH purpose_risk AS (
    SELECT 
        l.loan_purpose,
        COUNT(l.loan_id) AS total_loans,
        SUM(r.default_flag) AS total_defaults,
        ROUND(AVG(r.default_flag) * 100.0, 2) AS default_rate_pct
    FROM loans l
    JOIN repayment_status r ON l.loan_id = r.loan_id
    GROUP BY l.loan_purpose
)
SELECT 
    loan_purpose,
    total_loans,
    total_defaults,
    default_rate_pct,
    DENSE_RANK() OVER (ORDER BY default_rate_pct DESC) AS risk_rank
FROM purpose_risk;

-- query 10: window function rank to rank employment length bands by default risk
WITH employment_risk AS (
    SELECT 
        CASE 
            WHEN b.employment_length < 2 THEN '<2 years'
            WHEN b.employment_length BETWEEN 2 AND 5 THEN '2-5 years'
            WHEN b.employment_length BETWEEN 6 AND 10 THEN '6-10 years'
            ELSE '>10 years'
        END AS employment_band,
        COUNT(l.loan_id) AS total_loans,
        ROUND(AVG(r.default_flag) * 100.0, 2) AS default_rate_pct
    FROM borrowers b
    JOIN loans l ON b.borrower_id = l.borrower_id
    JOIN repayment_status r ON l.loan_id = r.loan_id
    GROUP BY employment_band
)
SELECT 
    employment_band,
    total_loans,
    default_rate_pct,
    RANK() OVER (ORDER BY default_rate_pct DESC) AS risk_rank
FROM employment_risk;

-- query 11: running total of exposure and default counts using window function
SELECT 
    l.loan_id,
    l.loan_purpose,
    l.loan_amount,
    r.default_flag,
    SUM(l.loan_amount) OVER (ORDER BY l.loan_id) AS running_total_exposure,
    SUM(r.default_flag) OVER (ORDER BY l.loan_id) AS running_total_defaults
FROM loans l
JOIN repayment_status r ON l.loan_id = r.loan_id
LIMIT 15;

-- query 12: window function avg over partition to calculate deviation from purpose average
SELECT 
    l.loan_id,
    l.borrower_id,
    l.loan_purpose,
    l.loan_amount,
    ROUND(AVG(l.loan_amount) OVER (PARTITION BY l.loan_purpose), 2) AS avg_purpose_amount,
    ROUND(l.loan_amount - AVG(l.loan_amount) OVER (PARTITION BY l.loan_purpose), 2) AS diff_from_purpose_avg
FROM loans l
LIMIT 15;

-- query 13: identifying defaulters with multiple existing loans and below average income
SELECT 
    b.borrower_id,
    b.age,
    b.income,
    l.loan_purpose,
    l.loan_amount,
    l.existing_loans_count,
    r.missed_payments_count
FROM borrowers b
JOIN loans l ON b.borrower_id = l.borrower_id
JOIN repayment_status r ON l.loan_id = r.loan_id
WHERE r.default_flag = 1 
  AND l.existing_loans_count > 1
  AND b.income < (SELECT AVG(income) FROM borrowers)
ORDER BY b.income ASC
LIMIT 15;

-- query 14: default metrics broken down by number of existing loans
SELECT 
    l.existing_loans_count,
    COUNT(l.loan_id) AS total_loans,
    SUM(r.default_flag) AS total_defaults,
    ROUND(AVG(r.default_flag) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(l.loan_amount), 2) AS avg_loan_amount
FROM loans l
JOIN repayment_status r ON l.loan_id = r.loan_id
GROUP BY l.existing_loans_count
ORDER BY l.existing_loans_count ASC;

-- query 15: default risk analysis across loan size quartiles using ntile window function
WITH borrower_summary AS (
    SELECT 
        b.borrower_id,
        l.loan_amount,
        r.default_flag
    FROM borrowers b
    JOIN loans l ON b.borrower_id = l.borrower_id
    JOIN repayment_status r ON l.loan_id = r.loan_id
),
risk_quartiles AS (
    SELECT 
        borrower_id,
        loan_amount,
        default_flag,
        NTILE(4) OVER (ORDER BY loan_amount DESC) AS loan_size_quartile
    FROM borrower_summary
)
SELECT 
    loan_size_quartile,
    COUNT(*) AS total_loans,
    SUM(default_flag) AS total_defaults,
    ROUND(AVG(default_flag) * 100.0, 2) AS default_rate_pct
FROM risk_quartiles
GROUP BY loan_size_quartile
ORDER BY loan_size_quartile ASC;
