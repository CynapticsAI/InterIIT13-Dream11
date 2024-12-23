import React from 'react'
import {getAllPlayers,getAllTeams} from '@/api'
import AddMatchPage from '../../components/AddMatchPage';
export default function page() {
  return (
    <div className='!overflow-hidden'><AddMatchPage /></div>
  )
}
