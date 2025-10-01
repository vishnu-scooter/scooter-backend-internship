from pydantic import BaseModel
from typing import Optional, List, Dict, Union
from datetime import datetime

# Pydantic model for data validation
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    is_available: bool = True

class EmploymentGaps(BaseModel):
    has_gaps: bool
    duration: str

class CompanyHistory(BaseModel):
    company_name: str
    position: str
    start_date: str
    end_date: str
    duration_months: int
    is_current: bool = False

class BasicInformation(BaseModel):
    full_name: str
    current_location: str
    open_to_relocation: bool
    phone_number: str
    linkedin_url: str
    email: str
    specific_phone_number: Optional[str] = None
    notice_period: str
    current_ctc: Dict[str, Union[str, float]]
    expected_ctc: Dict[str, Union[str, float]]

class CareerOverview(BaseModel):
    total_years_experience: float
    years_sales_experience: float
    average_tenure_per_role: float
    employment_gaps: EmploymentGaps
    promotion_history: bool
    company_history: List[CompanyHistory]

class SalesContext(BaseModel):
    sales_type: str
    sales_motion: str
    industries_sold_into: List[str]
    regions_sold_into: List[str]
    buyer_personas: List[str]

class RoleProcessExposure(BaseModel):
    sales_role_type: str
    position_level: str
    sales_stages_owned: List[str]
    average_deal_size: str
    sales_cycle_length: str
    own_quota: bool
    quota_ownership: List[str]
    quota_attainment: str

class ToolsPlatforms(BaseModel):
    crm_tools: List[str]
    sales_tools: List[str]

class ResumeData(BaseModel):
    basic_information: BasicInformation
    career_overview: CareerOverview
    sales_context: SalesContext
    role_process_exposure: RoleProcessExposure
    tools_platforms: ToolsPlatforms

##