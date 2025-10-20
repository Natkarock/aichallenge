from dataclasses import dataclass
from typing import Optional, Dict, Any
@dataclass
class DesignBrief:
    client_name:str; apartment_area:str; rooms:str; style:str; color_palette:str; materials:str; furniture:str; lighting:str; budget:str; special_requests:str; final_design_brief:str
    @staticmethod
    def from_dict(d:Dict[str,Any])->'DesignBrief':
        return DesignBrief(d.get('client_name',''),d.get('apartment_area',''),d.get('rooms',''),d.get('style',''),d.get('color_palette',''),d.get('materials',''),d.get('furniture',''),d.get('lighting',''),d.get('budget',''),d.get('special_requests','нет'),d.get('final_design_brief',''))
    def pretty_print(self,heading:Optional[str]=None)->None:
        if heading: print(heading)
        print('\n📐 Итоговое техническое задание (дизайн-бриф):\n')
        rows=[('Имя клиента',self.client_name),('Площадь',self.apartment_area),('Комнаты и назначение',self.rooms),('Стиль',self.style),('Цветовая палитра',self.color_palette),('Материалы и фактуры',self.materials),('Мебель и декор',self.furniture),('Освещение',self.lighting),('Бюджет',self.budget),('Особые пожелания',self.special_requests or 'нет')]
        for k,v in rows: print(f"{k}: {v}")
        print('\n— — —\n'); print(self.final_design_brief); print()
@dataclass
class DesignerTurn:
    mode:str; message:str; brief:Optional[DesignBrief]
    @staticmethod
    def from_dict(d:Dict[str,Any])->'DesignerTurn':
        b=d.get('brief');
        return DesignerTurn(d.get('mode','ask'),(d.get('message') or '').strip(),DesignBrief.from_dict(b) if isinstance(b,dict) else None)
