"""
Scenario Generator — Deterministic expense report and violation generation.

Produces realistic expense reports with seeded randomness for reproducibility.
Each task difficulty has its own scenario configuration.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from models import ExpenseItem, ExpenseReport, PolicyRule


# ── Realistic data pools ──────────────────────────────────────────────────

EMPLOYEE_NAMES = [
    "Sarah Chen", "Marcus Johnson", "Priya Patel", "James O'Brien",
    "Yuki Tanaka", "Carlos Mendez", "Elena Volkov", "David Kim",
    "Aisha Mohammed", "Robert Taylor", "Sofia Rossi", "Wei Zhang",
    "Hannah Fischer", "Miguel Santos", "Angela Park",
]

DEPARTMENTS = [
    "Engineering", "Sales", "Marketing", "Finance", "HR",
    "Product", "Operations", "Legal", "Customer Success", "Research",
]

MEAL_VENDORS = [
    "Olive Garden", "The Capital Grille", "Chipotle", "Subway",
    "Ruth's Chris Steak House", "Panera Bread", "McDonald's",
    "Nobu Restaurant", "Red Lobster", "Starbucks",
    "Local Diner", "Hotel Restaurant", "Airport Bistro",
]

TRAVEL_VENDORS = [
    "United Airlines", "Delta Airlines", "American Airlines",
    "Southwest Airlines", "Uber", "Lyft", "Enterprise Rent-A-Car",
    "Hertz", "National Car Rental", "Amtrak",
]

LODGING_VENDORS = [
    "Marriott", "Hilton", "Hyatt", "Holiday Inn", "Hampton Inn",
    "Westin", "Courtyard by Marriott", "Four Seasons", "Ritz-Carlton",
    "Best Western", "Airbnb",
]

OFFICE_VENDORS = [
    "Staples", "Office Depot", "Amazon Business", "Best Buy",
    "Apple Store", "Microsoft Store", "Newegg",
]

ENTERTAINMENT_VENDORS = [
    "AMC Theaters", "Topgolf", "Dave & Buster's", "Broadway Tickets",
    "Concert Hall", "Spa Resort", "Country Club",
]

SUSPICIOUS_VENDORS = [
    "Cash Payment", "Personal Services LLC", "Gift Shop Emporium",
    "Luxury Boutique", "Self LLC", "Family Restaurant (personal)",
]

BUSINESS_PURPOSES = [
    "Client meeting in Chicago", "Q4 sales conference",
    "Team offsite planning session", "Customer onboarding visit",
    "Annual industry conference", "Partner negotiation trip",
    "Regional training workshop", "Product launch event",
    "Board meeting preparation", "Recruitment fair",
]


# ── Violation types and ground truth ───────────────────────────────────────

@dataclass
class ViolationAnnotation:
    """Ground truth annotation for a violation."""
    item_id: str
    violation_type: str
    explanation: str
    severity: float = 1.0  # 0.0-1.0 how obvious the violation is


@dataclass
class ReportAnnotation:
    """Ground truth annotation for a complete report."""
    report_id: str
    should_reject: bool
    violations: List[ViolationAnnotation] = field(default_factory=list)
    clean_items: List[str] = field(default_factory=list)  # item_ids that are clean


@dataclass
class ScenarioConfig:
    """Configuration for a task scenario."""
    task_name: str
    num_reports: int
    items_per_report: Tuple[int, int]  # (min, max)
    violation_rate: float  # proportion of items that are violations
    max_steps: int
    policy_rules: List[PolicyRule] = field(default_factory=list)
    # Hard-mode features
    allow_split_transactions: bool = False
    allow_duplicates_across_reports: bool = False
    allow_vendor_disguises: bool = False
    allow_personal_expenses: bool = False


# ── Policy rule sets ──────────────────────────────────────────────────────

BASIC_POLICY = [
    PolicyRule(rule_id="P1", category="meals", description="Meals must not exceed $75 per person per meal.", limit=75.0),
    PolicyRule(rule_id="P2", category="all", description="All expenses over $25 must have a receipt attached.", limit=25.0),
    PolicyRule(rule_id="P3", category="lodging", description="Hotel stays must not exceed $250 per night.", limit=250.0),
    PolicyRule(rule_id="P4", category="travel", description="Economy class flights only; max $800 per domestic flight.", limit=800.0),
    PolicyRule(rule_id="P5", category="all", description="Expenses must be incurred on business days (Mon-Fri) unless pre-approved travel.", limit=None),
]

STANDARD_POLICY = BASIC_POLICY + [
    PolicyRule(rule_id="P6", category="entertainment", description="Entertainment expenses require prior manager approval and must not exceed $150 per event.", limit=150.0),
    PolicyRule(rule_id="P7", category="office_supplies", description="Office supply purchases above $200 require purchasing department approval.", limit=200.0),
    PolicyRule(rule_id="P8", category="meals", description="Alcohol is not reimbursable. Meals with alcohol must have the alcohol amount deducted.", limit=None),
    PolicyRule(rule_id="P9", category="all", description="Expenses must be submitted within 30 days of incurrence.", limit=None),
    PolicyRule(rule_id="P10", category="all", description="Duplicate submissions of the same expense are prohibited.", limit=None),
]

FORENSIC_POLICY = STANDARD_POLICY + [
    PolicyRule(rule_id="P11", category="meals", description="Group meals must list all attendees. Cost per person must not exceed $75.", limit=75.0),
    PolicyRule(rule_id="P12", category="all", description="Single transactions must not be split to stay under approval thresholds.", limit=None),
    PolicyRule(rule_id="P13", category="travel", description="Personal travel days embedded in business trips are not reimbursable.", limit=None),
    PolicyRule(rule_id="P14", category="all", description="Vendor must be a legitimate business. Cash payments over $50 require additional documentation.", limit=50.0),
    PolicyRule(rule_id="P15", category="lodging", description="Airbnb stays require pre-approval and must not exceed hotel-equivalent rates ($250/night).", limit=250.0),
    PolicyRule(rule_id="P16", category="all", description="Department-specific limits: Engineering $5000/month, Sales $8000/month, all others $3000/month.", limit=None),
    PolicyRule(rule_id="P17", category="entertainment", description="No entertainment expenses on weekends unless part of a conference or client event.", limit=None),
]


# ── Scenario generation ───────────────────────────────────────────────────

def _generate_clean_item(
    rng: random.Random,
    item_idx: int,
    date: str,
    policy: List[PolicyRule],
) -> ExpenseItem:
    """Generate a legitimate, policy-compliant expense item."""
    category = rng.choice(["meals", "travel", "lodging", "office_supplies", "entertainment"])

    if category == "meals":
        vendor = rng.choice(MEAL_VENDORS[:8])  # avoid suspicious ones
        amount = round(rng.uniform(12.0, 65.0), 2)
        desc = rng.choice([
            "Team lunch with 3 colleagues",
            "Client dinner meeting",
            "Working lunch during conference",
            "Breakfast at hotel",
        ])
        receipt_desc = f"Receipt from {vendor}: {desc}. Total ${amount:.2f}. Visa ending 4521."
    elif category == "travel":
        vendor = rng.choice(TRAVEL_VENDORS)
        amount = round(rng.uniform(25.0, 700.0), 2)
        desc = rng.choice([
            "Flight to client site",
            "Uber to conference center",
            "Rental car for site visits",
            "Train ticket to NYC office",
        ])
        receipt_desc = f"Booking confirmation: {vendor}. Amount: ${amount:.2f}. Confirmation #{''.join(rng.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))}."
    elif category == "lodging":
        vendor = rng.choice(LODGING_VENDORS[:8])
        amount = round(rng.uniform(120.0, 240.0), 2)
        desc = rng.choice([
            "Hotel stay for client meeting",
            "Conference hotel - 1 night",
            "Hotel near customer office",
        ])
        receipt_desc = f"Hotel folio: {vendor}. Room rate: ${amount:.2f}/night. Guest: Employee. Check-in: {date}."
    elif category == "office_supplies":
        vendor = rng.choice(OFFICE_VENDORS[:4])
        amount = round(rng.uniform(15.0, 180.0), 2)
        desc = rng.choice([
            "Printer paper and toner",
            "USB-C cables and adapters",
            "Whiteboard markers and erasers",
            "Desk organizer",
        ])
        receipt_desc = f"Order from {vendor}: {desc}. Total ${amount:.2f}. Order #{''.join(rng.choices('0123456789', k=8))}."
    else:  # entertainment
        vendor = rng.choice(ENTERTAINMENT_VENDORS[:3])
        amount = round(rng.uniform(30.0, 130.0), 2)
        desc = rng.choice([
            "Team building event",
            "Client entertainment",
            "Team celebration dinner add-on",
        ])
        receipt_desc = f"Receipt: {vendor}. Event for team outing. Total: ${amount:.2f}."

    return ExpenseItem(
        item_id=f"EXP-{item_idx:03d}",
        date=date,
        category=category,
        vendor=vendor,
        amount=amount,
        currency="USD",
        description=desc,
        receipt_present=True,
        receipt_description=receipt_desc,
    )


def _generate_violation_item(
    rng: random.Random,
    item_idx: int,
    date: str,
    violation_type: str,
    policy: List[PolicyRule],
) -> Tuple[ExpenseItem, ViolationAnnotation]:
    """Generate an expense item that violates a specific policy."""

    if violation_type == "over_limit":
        category = rng.choice(["meals", "lodging", "travel"])
        if category == "meals":
            vendor = rng.choice(MEAL_VENDORS)
            amount = round(rng.uniform(85.0, 280.0), 2)
            desc = "Business dinner with client"
            receipt_desc = f"Receipt from {vendor}: Fine dining. Total ${amount:.2f} (includes wine pairing). Amex ending 9012."
            explanation = f"Meal expense ${amount:.2f} exceeds $75 per-person limit (P1)"
        elif category == "lodging":
            vendor = rng.choice(["Four Seasons", "Ritz-Carlton", "Waldorf Astoria"])
            amount = round(rng.uniform(310.0, 650.0), 2)
            desc = "Hotel stay for conference"
            receipt_desc = f"Hotel folio: {vendor}. Suite rate: ${amount:.2f}/night. Minibar, spa charges included."
            explanation = f"Hotel rate ${amount:.2f} exceeds $250/night limit (P3)"
        else:
            vendor = rng.choice(TRAVEL_VENDORS[:3])
            amount = round(rng.uniform(900.0, 2500.0), 2)
            desc = "Flight to client meeting"
            receipt_desc = f"E-ticket: {vendor}. Business class seat. Total: ${amount:.2f}."
            explanation = f"Flight cost ${amount:.2f} exceeds $800 limit; business class not permitted (P4)"

    elif violation_type == "missing_receipt":
        category = rng.choice(["meals", "travel", "office_supplies"])
        vendor = rng.choice(MEAL_VENDORS + TRAVEL_VENDORS)
        amount = round(rng.uniform(35.0, 200.0), 2)
        desc = rng.choice(["Business lunch", "Taxi to airport", "Office supplies"])
        receipt_desc = ""
        explanation = f"No receipt for ${amount:.2f} expense over $25 threshold (P2)"
        return (
            ExpenseItem(
                item_id=f"EXP-{item_idx:03d}",
                date=date,
                category=category,
                vendor=vendor,
                amount=amount,
                currency="USD",
                description=desc,
                receipt_present=False,
                receipt_description="",
            ),
            ViolationAnnotation(
                item_id=f"EXP-{item_idx:03d}",
                violation_type="missing_receipt",
                explanation=explanation,
                severity=0.9,
            ),
        )

    elif violation_type == "wrong_category":
        # E.g., a meal categorized as office supplies
        vendor = rng.choice(MEAL_VENDORS)
        amount = round(rng.uniform(30.0, 60.0), 2)
        desc = "Team supplies" 
        receipt_desc = f"Receipt from {vendor}: Lunch for 4. Total ${amount:.2f}."
        category = "office_supplies"
        explanation = f"Item from {vendor} is a restaurant meal but categorized as office_supplies (wrong category)"
        return (
            ExpenseItem(
                item_id=f"EXP-{item_idx:03d}",
                date=date,
                category=category,
                vendor=vendor,
                amount=amount,
                currency="USD",
                description=desc,
                receipt_present=True,
                receipt_description=receipt_desc,
            ),
            ViolationAnnotation(
                item_id=f"EXP-{item_idx:03d}",
                violation_type="wrong_category",
                explanation=explanation,
                severity=0.7,
            ),
        )

    elif violation_type == "policy_violation":
        # Weekend expense for non-travel
        category = "entertainment"
        vendor = rng.choice(ENTERTAINMENT_VENDORS)
        amount = round(rng.uniform(50.0, 200.0), 2)
        # Use a weekend date
        desc = "Weekend entertainment"
        receipt_desc = f"Receipt: {vendor}. Saturday evening event. Total: ${amount:.2f}."
        explanation = f"Entertainment expense on weekend without pre-approval (P5/P17)"
        return (
            ExpenseItem(
                item_id=f"EXP-{item_idx:03d}",
                date=date,
                category=category,
                vendor=vendor,
                amount=amount,
                currency="USD",
                description=desc,
                receipt_present=True,
                receipt_description=receipt_desc,
            ),
            ViolationAnnotation(
                item_id=f"EXP-{item_idx:03d}",
                violation_type="policy_violation",
                explanation=explanation,
                severity=0.6,
            ),
        )

    elif violation_type == "suspicious_vendor":
        category = rng.choice(["meals", "misc"])
        vendor = rng.choice(SUSPICIOUS_VENDORS)
        amount = round(rng.uniform(60.0, 300.0), 2)
        desc = "Business supplies"
        receipt_desc = f"Handwritten receipt: {vendor}. Cash payment. ${amount:.2f}."
        explanation = f"Suspicious vendor '{vendor}' with cash payment — possible personal expense (P14)"
        return (
            ExpenseItem(
                item_id=f"EXP-{item_idx:03d}",
                date=date,
                category=category,
                vendor=vendor,
                amount=amount,
                currency="USD",
                description=desc,
                receipt_present=True,
                receipt_description=receipt_desc,
            ),
            ViolationAnnotation(
                item_id=f"EXP-{item_idx:03d}",
                violation_type="suspicious_vendor",
                explanation=explanation,
                severity=0.5,
            ),
        )

    elif violation_type == "personal_expense":
        category = rng.choice(["entertainment", "meals"])
        vendor = rng.choice(["Spa Resort", "Gift Shop Emporium", "Luxury Boutique"])
        amount = round(rng.uniform(80.0, 350.0), 2)
        desc = rng.choice(["Supplies for home office", "Gift for team", "Wellness session"])
        receipt_desc = f"Receipt: {vendor}. Personal spa treatment / gift purchase. ${amount:.2f}."
        explanation = f"Personal expense disguised as business: {vendor} ({desc})"
        return (
            ExpenseItem(
                item_id=f"EXP-{item_idx:03d}",
                date=date,
                category=category,
                vendor=vendor,
                amount=amount,
                currency="USD",
                description=desc,
                receipt_present=True,
                receipt_description=receipt_desc,
            ),
            ViolationAnnotation(
                item_id=f"EXP-{item_idx:03d}",
                violation_type="personal_expense",
                explanation=explanation,
                severity=0.4,
            ),
        )

    elif violation_type == "split_transaction":
        # Two items from same vendor on same day, both just under limit
        category = "office_supplies"
        vendor = rng.choice(OFFICE_VENDORS)
        amount = round(rng.uniform(180.0, 198.0), 2)
        desc = f"Office equipment purchase (part 1)"
        receipt_desc = f"Order from {vendor}: Monitors. ${amount:.2f}. Same-day order split."
        explanation = f"Appears to be a split transaction from {vendor} to stay under $200 approval threshold (P12)"
        return (
            ExpenseItem(
                item_id=f"EXP-{item_idx:03d}",
                date=date,
                category=category,
                vendor=vendor,
                amount=amount,
                currency="USD",
                description=desc,
                receipt_present=True,
                receipt_description=receipt_desc,
            ),
            ViolationAnnotation(
                item_id=f"EXP-{item_idx:03d}",
                violation_type="split_transaction",
                explanation=explanation,
                severity=0.3,
            ),
        )

    elif violation_type == "duplicate":
        category = rng.choice(["meals", "travel"])
        vendor = rng.choice(MEAL_VENDORS + TRAVEL_VENDORS)
        amount = round(rng.uniform(40.0, 300.0), 2)
        desc = "Business meal / travel"
        receipt_desc = f"Receipt: {vendor}. ${amount:.2f}. (Duplicate of previous submission)"
        explanation = f"Duplicate submission of ${amount:.2f} expense at {vendor} (P10)"
        return (
            ExpenseItem(
                item_id=f"EXP-{item_idx:03d}",
                date=date,
                category=category,
                vendor=vendor,
                amount=amount,
                currency="USD",
                description=desc,
                receipt_present=True,
                receipt_description=receipt_desc,
            ),
            ViolationAnnotation(
                item_id=f"EXP-{item_idx:03d}",
                violation_type="duplicate",
                explanation=explanation,
                severity=0.6,
            ),
        )
    else:
        # Fallback: over_limit meal
        category = "meals"
        vendor = "Expensive Restaurant"
        amount = round(rng.uniform(100.0, 250.0), 2)
        desc = "Business dinner"
        receipt_desc = f"Receipt: {vendor}. Fine dining. ${amount:.2f}."
        explanation = f"Meal ${amount:.2f} exceeds $75 limit (P1)"

    return (
        ExpenseItem(
            item_id=f"EXP-{item_idx:03d}",
            date=date,
            category=category,
            vendor=vendor,
            amount=amount,
            currency="USD",
            description=desc,
            receipt_present=True,
            receipt_description=receipt_desc,
        ),
        ViolationAnnotation(
            item_id=f"EXP-{item_idx:03d}",
            violation_type=violation_type,
            explanation=explanation,
            severity=0.8,
        ),
    )


def _generate_dates(rng: random.Random, count: int, allow_weekends: bool = False) -> List[str]:
    """Generate realistic business-day dates."""
    year = 2025
    month = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    dates = []
    day = rng.randint(1, 20)
    for _ in range(count):
        # Simple date generation
        import datetime
        d = datetime.date(year, month, min(day, 28))
        if not allow_weekends:
            # Shift weekends to Monday
            while d.weekday() >= 5:
                d += datetime.timedelta(days=1)
        dates.append(d.isoformat())
        day += rng.randint(1, 3)
        if day > 28:
            day = 1
            month = min(month + 1, 12)
    return dates


def generate_scenario(
    task_name: str,
    seed: int = 42,
) -> Tuple[List[ExpenseReport], List[ReportAnnotation], List[PolicyRule], int]:
    """
    Generate a complete scenario for a given task.

    Returns:
        (reports, annotations, policy_rules, max_steps)
    """
    rng = random.Random(seed)

    if task_name == "basic_audit":
        config = ScenarioConfig(
            task_name="basic_audit",
            num_reports=3,
            items_per_report=(3, 5),
            violation_rate=0.35,
            max_steps=30,
            policy_rules=BASIC_POLICY,
        )
        easy_violation_types = ["over_limit", "missing_receipt"]
    elif task_name == "standard_audit":
        config = ScenarioConfig(
            task_name="standard_audit",
            num_reports=5,
            items_per_report=(5, 8),
            violation_rate=0.30,
            max_steps=50,
            policy_rules=STANDARD_POLICY,
            allow_duplicates_across_reports=True,
        )
        easy_violation_types = [
            "over_limit", "missing_receipt", "wrong_category",
            "policy_violation", "duplicate",
        ]
    elif task_name == "forensic_audit":
        config = ScenarioConfig(
            task_name="forensic_audit",
            num_reports=8,
            items_per_report=(6, 12),
            violation_rate=0.25,
            max_steps=80,
            policy_rules=FORENSIC_POLICY,
            allow_split_transactions=True,
            allow_duplicates_across_reports=True,
            allow_vendor_disguises=True,
            allow_personal_expenses=True,
        )
        easy_violation_types = [
            "over_limit", "missing_receipt", "wrong_category",
            "policy_violation", "duplicate", "suspicious_vendor",
            "personal_expense", "split_transaction",
        ]
    else:
        raise ValueError(f"Unknown task: {task_name}")

    reports: List[ExpenseReport] = []
    annotations: List[ReportAnnotation] = []
    global_item_idx = 0

    for report_idx in range(config.num_reports):
        num_items = rng.randint(*config.items_per_report)
        employee = rng.choice(EMPLOYEE_NAMES)
        department = rng.choice(DEPARTMENTS)
        purpose = rng.choice(BUSINESS_PURPOSES)
        dates = _generate_dates(rng, num_items)

        items: List[ExpenseItem] = []
        violations: List[ViolationAnnotation] = []
        clean_item_ids: List[str] = []

        # Decide which items are violations
        num_violations = max(1, int(num_items * config.violation_rate))
        violation_indices: Set[int] = set(rng.sample(range(num_items), min(num_violations, num_items)))

        for i in range(num_items):
            if i in violation_indices:
                vtype = rng.choice(easy_violation_types)
                item, annotation = _generate_violation_item(
                    rng, global_item_idx, dates[i], vtype, config.policy_rules,
                )
                items.append(item)
                violations.append(annotation)
            else:
                item = _generate_clean_item(rng, global_item_idx, dates[i], config.policy_rules)
                items.append(item)
                clean_item_ids.append(item.item_id)
            global_item_idx += 1

        total_amount = round(sum(it.amount for it in items), 2)
        report = ExpenseReport(
            report_id=f"RPT-{report_idx + 1:03d}",
            employee_name=employee,
            employee_id=f"EMP-{rng.randint(1000, 9999)}",
            department=department,
            submission_date=dates[0],
            business_purpose=purpose,
            expenses=items,
            total_amount=total_amount,
        )
        reports.append(report)

        should_reject = len(violations) > 0
        annotations.append(ReportAnnotation(
            report_id=report.report_id,
            should_reject=should_reject,
            violations=violations,
            clean_items=clean_item_ids,
        ))

    return reports, annotations, config.policy_rules, config.max_steps
